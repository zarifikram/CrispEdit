from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.func import functional_call, hessian
from tqdm import tqdm, trange
from functools import partial

SEED = 69
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

LR_PRE = 0.1
LR_FT = 1e-3
EPOCHS_PRE = 29
EPOCHS_FT = 100
BATCH_SIZE = 256

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "mps")

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class ProjectedSGDFlatten(optim.Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(ProjectedSGDFlatten, self).__init__(params, defaults)

        for group in self.param_groups:
            if 'P_null' not in group:
                raise ValueError("Each param group must contain 'P_null' projectors.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            P_null = group['P_null'].to(DEVICE)
            
            weight_param = group['params'][0]
            bias_param = group['params'][1]

            if weight_param.grad is None:
                continue

            grad_W = weight_param.grad.data

            grad_combined = grad_W.view(-1)

            grad_proj_combined = P_null @ grad_combined

            grad_W_proj = grad_proj_combined.view_as(grad_W)

            weight_param.add_(grad_W_proj, alpha=-lr)
        return loss

class ProjectedSGD(optim.Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(ProjectedSGD, self).__init__(params, defaults)

        # Check if projectors are provided in each group
        for group in self.param_groups:
            if 'P_A_null' not in group and 'P_B_null' not in group:
                raise ValueError("Each param group must contain 'P_A_null' and 'P_B_null' projectors.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            P_A_null = group['P_A_null'].to(DEVICE)

            if "P_B_null" in group:
                P_B_null = group['P_B_null'].to(DEVICE)
            
            weight_param = group['params'][0]
            bias_param = group['params'][1]

            if weight_param.grad is None:
                continue

            # Get gradients
            grad_W = weight_param.grad.data
            
            grad_W_proj = grad_W @ P_A_null
            
            if "P_B_null" in group:
                grad_W_proj = P_B_null @ grad_W_proj

            weight_param.add_(grad_W_proj, alpha=-lr)

        return loss

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

def calculate_updated_kfac_nofac_for_layer(model, loader, criterion, device, layer):
    model.eval()
    
    in_dim = layer.in_features
    out_dim = layer.out_features

    kron = torch.zeros((out_dim * (in_dim), out_dim * (in_dim)), device=device)
    N = 0

    for inputs, targets in tqdm(loader, desc="Calculating Updated K-FAC"):
        inputs, targets = inputs.to(device), targets.to(device)
        B = torch.zeros((out_dim, out_dim), device=device)
        for inp, target in zip(inputs, targets):
            layer_input = calculate_inner_layer_input(model, layer, inp.unsqueeze(0))
            A = layer_input.T @ layer_input  # [D_in, 1] @ [1, D_in]
            jacobian_i, probs_i = compute_jacobian_and_probs(model, layer, inp.unsqueeze(0))
            j_transpose_p = jacobian_i.T @ probs_i.unsqueeze(1) 
            jacobian_i_diag_p_jacobian = (jacobian_i.T * probs_i) @ jacobian_i
            B = (jacobian_i_diag_p_jacobian - j_transpose_p @ j_transpose_p.T)
            kron += torch.kron(B, A)
            N += 1

    kron /= N
    return kron

def calculate_inner_layer_input(model, layer, input_data):
    captured_input = None
    captured_grad_output = None

    def save_input_hook(module, input_tuple, output):
        nonlocal captured_input
        captured_input = input_tuple[0]
        if not captured_input.is_leaf:
             captured_input = captured_input.detach()

    h_input = layer.register_forward_hook(save_input_hook)
    model.zero_grad()
    _ = model(input_data)

    if captured_input.dim() > 2:
        captured_input_flat = captured_input.view(-1, captured_input.size(-1))
    else:
        captured_input_flat = captured_input
    h_input.remove()
    return captured_input_flat

def calculate_updated_kfac_factors_for_layer(model, loader, criterion, device, layer):
    model.eval()
    
    in_dim = layer.in_features
    out_dim = layer.out_features

    A = torch.zeros((in_dim, in_dim), device=device)
    B = torch.zeros((out_dim, out_dim), device=device)
    # kron = torch.zeros((out_dim * (in_dim), out_dim * (in_dim)), device=device)
    N = 0

    for inputs, targets in tqdm(loader, desc="Calculating Updated K-FAC"):
        inputs, targets = inputs.to(device), targets.to(device)
        for inp, target in zip(inputs, targets):
            layer_input = calculate_inner_layer_input(model, layer, inp.unsqueeze(0))
            A_i = layer_input.T @ layer_input

            jacobian_i, probs_i = compute_jacobian_and_probs(model, layer, inp.unsqueeze(0))
            j_transpose_p = jacobian_i.T @ probs_i.unsqueeze(1) 
            jacobian_i_diag_p_jacobian = (jacobian_i.T * probs_i) @ jacobian_i
            B_i = jacobian_i_diag_p_jacobian - j_transpose_p @ j_transpose_p.T

            A += A_i
            B += B_i
            # kron += torch.kron(B_i, A_i)
            N += 1

    # kron /= N
    A /= N
    B /= N
    # print(f"Kron approximation relative error with updated factors: {torch.norm(kron - torch.kron(B, A)) / (torch.norm(kron) + 1e-10):.6f}")
    return A, B

def compute_jacobian_and_probs(model, layer, input_data):
    jacobian_rows = []
    pre_act_tensor = None
    
    # --- 1. Forward Hook ---
    def get_pre_activation(module, inp, output):
        nonlocal pre_act_tensor
        pre_act_tensor = output
        if pre_act_tensor.requires_grad:
            pre_act_tensor.retain_grad()
        else:
            print("Warning: The pre-activation tensor does not require gradients.")

    hook_handle = layer.register_forward_hook(get_pre_activation)

    model.eval()
    model.zero_grad()
    
    try:
        logits = model(input_data)
        probs = F.softmax(logits, dim=1)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        hook_handle.remove()
        return None
    
    if pre_act_tensor is None:
        print("Error: The forward hook did not capture the pre-activation tensor.")
        print("Are you sure you passed the correct layer?")
        hook_handle.remove()
        return None
    num_logits = logits.shape[-1]
    
    num_pre_activations = pre_act_tensor.view(1, -1).shape[1]

    for i in range(num_logits):
        model.zero_grad() # Zero gradients for a clean pass
        probs[0, i].backward(retain_graph=True)
        
        if pre_act_tensor.grad is not None:
            jacobian_row = pre_act_tensor.grad.view(-1).clone()
            jacobian_rows.append(jacobian_row)
            
            pre_act_tensor.grad.zero_()
        else:
            print(f"Warning: Gradient for pre_act_tensor was None on logit {i}.")
            jacobian_rows.append(torch.zeros(num_pre_activations))

    hook_handle.remove()

    if not jacobian_rows:
        print("Error: No gradients were collected.")
        return None
        
    jacobian = torch.stack(jacobian_rows)
    return jacobian, F.softmax(logits, dim=1).squeeze()

def calculate_kfac_factors_for_layer(model, loader, criterion, device, layer):
    model.eval()
    
    in_dim = layer.in_features
    out_dim = layer.out_features
    
    A = torch.zeros((in_dim, in_dim), device=device)
    B = torch.zeros((out_dim, out_dim), device=device)

    N = 0 

    captured_input = None
    captured_grad_output = None

    def save_input_hook(module, input_tuple, output):
        nonlocal captured_input
        captured_input = input_tuple[0]
        if not captured_input.is_leaf:
             captured_input = captured_input.detach()

    def save_grad_output_hook(module, grad_input, grad_output_tuple):
        nonlocal captured_grad_output
        captured_grad_output = grad_output_tuple[0]

    h_input = layer.register_forward_hook(save_input_hook)
    h_grad = layer.register_backward_hook(save_grad_output_hook)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if captured_input.dim() > 2:
            captured_input_flat = captured_input.view(-1, captured_input.size(-1))
        else:
            captured_input_flat = captured_input
            
        current_batch_size = captured_input_flat.size(0)
        
        A += captured_input_flat.T @ captured_input_flat # [D_in, B] @ [B, D_in]
        
        g = captured_grad_output # [B, D_out]
        B += g.t() @ g # [D_out, B] @ [B, D_out]
        
        N += batch_size

    h_input.remove()
    h_grad.remove()

    A /= N
    B /= N
    return A, B

def calculate_nparams(layer):
    n_params = 0
    for param in layer.parameters():
        n_params += param.numel()
    return n_params

def calculate_flattened_params(layer):
    return layer.weight.view(-1)

def calculate_model_output_dim(model, loader, device):
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            return outputs.shape[1]
    return None

def calculate_jacobian_matrix(model, loader, device, layer):
    nparams = calculate_flattened_params(layer).numel()
    output_dim = calculate_model_output_dim(model, loader, device)
    jacobian_matrix = torch.ones((output_dim, nparams), device=device)
    n_samples = 0
    for inputs, _ in tqdm(loader, desc="Calculating Jacobian"):
        inputs = inputs.to(device)
        for input_data in inputs:
            jacobian_matrix += calculate_jacobian_single_input(model, input_data.unsqueeze(0), device, layer)
            n_samples += 1

    return jacobian_matrix / n_samples

def calculate_jacobian_single_input(model, input_data, device, layer):
    model.eval()
    input_data = input_data.to(device)
    model.zero_grad()

    output = model(input_data)
    output_dim = output.shape[1]
    probs = F.softmax(output, dim=1).squeeze()
    params = list(layer.parameters())
    params = [p for p in params if p.requires_grad]
    
    jacobian = torch.zeros((output_dim, layer.weight.numel()), device=device)

    for k in range(output_dim):
        retain_graph = (k < output_dim)
        p_k = probs[k]

        grads_list = torch.autograd.grad(
            p_k,
            params, # Pass the list of original tensors
            retain_graph=retain_graph,
            allow_unused=True # Good to keep
        )        

        jacobian[k, :] = grads_list[0].view(-1)

    # jacobian_approx = torch.kron(jacobian_i, layer_input)

    return jacobian

def get_null_space_projector(K, null_threshold=1e-5):
    U, S, Vh = torch.linalg.svd(K)
    rank, _ = get_rank(S, percent=null_threshold)
    # rank, _ = get_rank_entropy(S)
    # rank = (S >= null_threshold).sum().item()
    U_hat = U[:, rank:]
    P_null = U_hat @ U_hat.t()
    return P_null

def get_null_space_projectors(A, B, null_threshold=1e-5):
    P_A_null = get_null_space_projector(A, null_threshold=null_threshold)
    P_B_null = get_null_space_projector(B, null_threshold=null_threshold)
    return P_A_null, P_B_null

def get_null_space_projector_kron_efficient(A, B, null_threshold=1e-5):
    Sa, Ua = torch.linalg.eigh(A) 
    Sb, Ub = torch.linalg.eigh(B)

    S = torch.kron(Sa, Sb)
    _, null_threshold = get_rank(S, percent=null_threshold)
    # _, null_threshold = get_rank_entropy(S)

    V_hat_null = []
    for i in range(Sa.numel()):
        for j in range(Sb.numel()):
            # if torch.abs(Sa[i] * Sb[j]) < null_threshold and torch.abs(Sa[i] * Sb[j]) > 1e-14:
            if torch.abs(Sa[i] * Sb[j]) < null_threshold:
                v_a = Ua[:, i]
                v_b = Ub[:, j]
                
                v_kron = torch.kron(v_a, v_b).unsqueeze(1)
                V_hat_null.append(v_kron)

    if not V_hat_null:
        dim = A.shape[0] * B.shape[0]
        return torch.zeros(dim, dim, dtype=A.dtype, device=A.device)

    V_hat = torch.cat(V_hat_null, dim=1)
    Q, _ = torch.linalg.qr(V_hat)
    P_null = Q @ Q.T
                
    return P_null

def get_null_space_projector_kron_corrected(A, B, null_threshold, model, loader, criterion, device, layer):
    model.eval()
    in_dim = layer.in_features
    out_dim = layer.out_features
    corrected_S = torch.zeros((in_dim, out_dim), device=device)

    

    U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
    U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)

    n_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        g_weight = layer.weight.grad.data # [D_out, D_in]
        corrected_S += (U_B.t() @ g_weight.T @ U_A).pow(2)

    n_samples += 1

    corrected_S = corrected_S.div(n_samples).T
    _, null_threshold = get_rank(corrected_S.flatten(), percent=null_threshold)

    eigen_vectors = []
    for i in range(len(S_A)):
        for j in range(len(S_B)):
            if corrected_S[i, j] < null_threshold:
                U_A_hat = U_A[:, i]
                U_B_hat = U_B[:, j]
                U_kron_hat = torch.kron(U_A_hat, U_B_hat).unsqueeze(1)
                eigen_vectors.append(U_kron_hat)
    U_kron_null = torch.cat(eigen_vectors, dim=1)
    print(f"Kronecker projector null space dimension: {U_kron_null.shape[1]}")
    P_kron_null = U_kron_null @ U_kron_null.t()
    return P_kron_null, corrected_S.flatten()

def get_linear_layer_hessian(model: nn.Module, 
                             loader: DataLoader, 
                             criterion: nn.Module, 
                             device: torch.device, 
                             layer: nn.Module):
    model_params_map = {p: name for name, p in model.named_parameters()}
    layer_weight_name = model_params_map.get(layer.weight, None)

    param_names = []
    constant_params = {}
    for name, param in model.named_parameters():
        param_names.append(name)
        if name != layer_weight_name:
            constant_params[name] = param.detach()

    buffers = dict(model.named_buffers())
    
    w_shape = layer.weight.shape
    w_numel = layer.weight.numel()
    
    n_params_layer = w_numel
    total_hessian = torch.zeros((n_params_layer, n_params_layer), device=device)
    num_samples = 0

    def compute_loss_flat(flat_layer_params, data_batch, target_batch):
        layer_weight = flat_layer_params.reshape(w_shape)
        
        params_dict = {
            **constant_params,
            layer_weight_name: layer_weight,
        }
        
        output = functional_call(model, (params_dict, buffers), (data_batch,))
        
        # 4. Compute the loss
        return criterion(output, target_batch)

    current_flat_params = layer.weight.flatten()

    model.eval() # Ensure model is in eval mode
    for data, target in tqdm(loader, desc="Calculating Hessian"):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]

        def compute_loss_for_batch(flat_params):
            return compute_loss_flat(flat_params, data, target)

        hessian_batch = hessian(compute_loss_for_batch)(current_flat_params)

        total_hessian += hessian_batch * batch_size
        num_samples += batch_size

        # remove cache 
        torch.cuda.empty_cache()

    return total_hessian / num_samples

def get_linear_layer_gauss_newton_matrix(model, loader, criterion, device, layer):
    nparams = calculate_flattened_params(layer).numel()
    G = torch.zeros((nparams, nparams), device=device)

    n_samples = 0
    for inputs, targets in tqdm(loader, desc="Calculating Gauss-Newton"):
        for input_data, target in zip(inputs, targets):
            input_data, target = input_data.to(device), target.to(device)
            model.eval()
            model.zero_grad()

            output = model(input_data.unsqueeze(0))
            probs = F.softmax(output, dim=1).squeeze()

            jacobian_i = calculate_jacobian_single_input(model, input_data.unsqueeze(0), device, layer)
            # G_i = j_i^T * (diag(p) - p p^T) * j_i
            p_diag = torch.diag(probs)
            p_outer = probs.unsqueeze(1) @ probs.unsqueeze(0)
            fisher_info = p_diag - p_outer
            G_i = jacobian_i.T @ fisher_info @ jacobian_i
            G += G_i 
            n_samples += 1

    G /= n_samples
    return G

def get_kronecker(A, B):
    K = torch.kron(B, A)
    return K

def get_rank(eigenvalues, percent=0.9):
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy

    k = torch.searchsorted(energy_ratio, percent).item() + 1  # +1 for 0-based index
    threshold = sorted_eigvals[k-1] if k - 1 < len(sorted_eigvals) else 0.0
    print(f"Selected rank k: {k}, Threshold eigenvalue: {threshold:.6f} for energy percent: {percent}")
    return k, threshold.item()

def get_rank_entropy(eigenvalues):
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    normalized_eigvals = sorted_eigvals / (torch.sum(sorted_eigvals) + 1e-10)
    entropy = -torch.sum(normalized_eigvals * torch.log(normalized_eigvals + 1e-10))
    rank = int(torch.exp(entropy).item()) + 1
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else 0.0
    print(f"GER Rank: {rank}, Threshold eigenvalue: {threshold:.6f} based on entropy.")
    return rank, threshold.item()

global jac_hes_jac

def calculate_relative_weight_difference(model, layer_name, original_parameter):
    current_parameter = dict(model.named_modules())[layer_name].weight
    weight_diff = torch.norm(current_parameter - original_parameter).item()
    relative_weight_diff = weight_diff / (torch.norm(original_parameter).item() + 1e-10)
    return relative_weight_diff

def fine_tune(model, opt, ft_dataloader_train, ft_dataloader_test, pt_dataloader_test, criterion, fn_get_optimizer_with_model, layer_name):
    ft_accs, ft_losses, pre_accs, pre_losses, weight_differences = [], [], [], [], []
    pre_acc, ft_acc = 0.0, 0.0
    progress_bar = trange(EPOCHS_FT, desc=f"PT acc {pre_acc:.2f}%, FT acc {ft_acc:.2f}%")
    parameter_of_interest = dict(model.named_modules())[layer_name].weight.detach().clone()
    for epoch in progress_bar:
        train_epoch(model, ft_dataloader_train, opt, criterion)
        ft_loss, ft_acc = test(model, ft_dataloader_test, criterion)
        pre_loss, pre_acc = test(model, pt_dataloader_test, criterion)
        relative_weight_difference = calculate_relative_weight_difference(model, layer_name, parameter_of_interest)
        if relative_weight_difference > 0.25:
            print(f"Recalculating optimizer at epoch {epoch+1} due to large weight change: {relative_weight_difference*100:.4f}%")
            opt = fn_get_optimizer_with_model(model)
            parameter_of_interest = dict(model.named_modules())[layer_name].weight.detach().clone()
        ft_losses.append(ft_loss)
        ft_accs.append(ft_acc)
        pre_losses.append(pre_loss)
        pre_accs.append(pre_acc)
        weight_differences.append(relative_weight_difference)
        progress_bar.set_description(f"PT acc {pre_acc:.2f}%, FT acc {ft_acc:.2f}% | Weight Diff: {relative_weight_difference*100:.4f}%")

    print(f"Fine-tuning completed. Final FT Acc: {ft_accs[-1]:.2f}%, Final Pre Acc: {pre_accs[-1]:.2f}%")
    return ft_accs, ft_losses, pre_accs, pre_losses, weight_differences

def get_pretrain_and_approx_data(train_data_percentage):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set_pre = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader_pre = DataLoader(train_set_pre, batch_size=BATCH_SIZE, shuffle=True)
    test_set_pre = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader_pre = DataLoader(test_set_pre, batch_size=BATCH_SIZE, shuffle=False)

    total_data = len(train_set_pre)
    approx_data_size = int(total_data * train_data_percentage)
    indices = torch.randperm(total_data)[:approx_data_size]
    approx_subset = torch.utils.data.Subset(train_set_pre, indices)
    approx_loader = DataLoader(approx_subset, batch_size=8, shuffle=True)
    return train_loader_pre, test_loader_pre, approx_loader

def pretrain_and_save(model, train_loader, test_loader, criterion):
    optimizer = optim.SGD(model.parameters(), lr=LR_PRE)
    best_state_dict = None
    best_acc = 0.0
    for epoch in trange(EPOCHS_PRE, desc="Pre-training"):
        train_epoch(model, train_loader, optimizer, criterion)
        train_loss, train_acc = test(model, train_loader, criterion)
        test_loss, test_acc = test(model, test_loader, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state_dict = model.state_dict()
        print(f"Epoch {epoch+1}/{EPOCHS_PRE} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # save the model
    torch.save({
        'model_state_dict': best_state_dict
    }, f'model_cache/lenet_epoch_{EPOCHS_PRE}.pth')

def load_pretrained_model(train_loader, test_loader, criterion):
    pretrain_path = f'model_cache/lenet_epoch_{EPOCHS_PRE}.pth'
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location=DEVICE)
        model = Lenet().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained model loaded.")
        test_loss, test_acc = test(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        return model
    else:
        print("No pre-trained model found.")
        pretrain_model = Lenet().to(DEVICE)
        pretrain_and_save(pretrain_model, train_loader, test_loader, criterion)
        return pretrain_model

def get_ft_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set_ft = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader_ft = DataLoader(train_set_ft, batch_size=BATCH_SIZE, shuffle=True)
    test_set_ft = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader_ft = DataLoader(test_set_ft, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader_ft, test_loader_ft

def turn_off_grad_except_layer(model, layer):
    for param in model.parameters():
        param.requires_grad = False
    for param in layer.parameters():
        # if param is bias also disable
        if param.dim() == 1:
            continue
        param.requires_grad = True


def plot_spectra(spectra, title='Spectra', filename='spectra.png'):
    # rank_0_7, th_0_7 = get_rank(spectra, percent=0.7)
    # rank_0_9, th_0_9 = get_rank(spectra, percent=0.9)
    # rank_0_95, th_0_95 = get_rank(spectra, percent=0.95)
    # rank_0_99, th_0_99 = get_rank(spectra, percent=0.99)
    # rank_effective, th_effective = get_rank_entropy(spectra)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(spectra)), spectra.detach().cpu().numpy(), s=10, label='Singular Values')
    # plt.yscale('log')
    # plt.scatter([rank_0_7-1], [th_0_7], color='red', label='70% Energy Rank', s=100)
    # plt.scatter([rank_0_9-1], [th_0_9], color='green', label='90% Energy Rank', s=100)
    # plt.scatter([rank_0_95-1], [th_0_95], color='blue', label='95% Energy Rank', s=100)
    # plt.scatter([rank_0_99-1], [th_0_99], color='purple', label='99% Energy Rank', s=100)
    # plt.scatter([rank_effective-1], [th_effective], color='orange', label='Effective Rank', s=100)

    # plt.title('' + title)
    # plt.xlabel('Index')
    # plt.ylabel('Value (log scale)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(filename)
    # plt.close()
    # do nothing
    pass

def calculate_sgd_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    return optim.SGD(layer.parameters(), lr=lr)

def calculate_alphaedit_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth') and try_load:
        checkpoint = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth', map_location=DEVICE)
        A, B = checkpoint['A'],  checkpoint['B']
    else:
        A, B = calculate_kfac_factors_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save({'A': A, 'B': B}, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth')

    plot_spectra(torch.linalg.svdvals(A), title='A Singular Values', filename=f'spectra/A_spectra_{train_data_percentage:.2f}_{layer_name}.png')
    plot_spectra(torch.linalg.svdvals(B), title='B Singular Values', filename=f'spectra/B_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    P_A_null = get_null_space_projector(A, null_threshold=null_threshold)

    return ProjectedSGD(
        [
            {'params': layer.parameters(), 'P_A_null': P_A_null}
        ],
        lr=lr
    )

def calculate_jigsaw_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth') and try_load:
        checkpoint = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth', map_location=DEVICE)
        A, B = checkpoint['A'],  checkpoint['B']
    else:
        A, B = calculate_kfac_factors_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save({'A': A, 'B': B}, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth')

    P_A_null, P_B_null = get_null_space_projectors(A, B, null_threshold=null_threshold)
    return ProjectedSGD(
        [
            {'params': layer.parameters(), 'P_A_null': P_A_null, 'P_B_null': P_B_null}
        ],
        lr=lr
    )

def calculate_hessian_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_hessian.pth') and try_load:
        hessian = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_hessian.pth', map_location=DEVICE)
    else:
        hessian = get_linear_layer_hessian(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save(hessian, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_hessian.pth')

    plot_spectra(torch.linalg.svdvals(hessian), title='Hessian Singular Values', filename=f'spectra/hessian_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    P_null = get_null_space_projector(hessian, null_threshold=null_threshold)
    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_gauss_newton_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_gn_hessian.pth') and try_load:
        gn_hessian = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_gn_hessian.pth', map_location=DEVICE)
    else:
        gn_hessian = get_linear_layer_gauss_newton_matrix(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save(gn_hessian, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_gn_hessian.pth')

    plot_spectra(torch.linalg.svdvals(gn_hessian), title='Gauss-Newton Hessian Singular Values', filename=f'spectra/gn_hessian_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    P_null = get_null_space_projector(gn_hessian, null_threshold=null_threshold)
    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_jacobian_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_jacobian.pth') and try_load:
        jacobian_matrix = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_jacobian.pth', map_location=DEVICE)
    else:
        jacobian_matrix = calculate_jacobian_matrix(model, approx_loader, DEVICE, layer)
        if try_load:
            torch.save(jacobian_matrix, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_jacobian.pth')

    plot_spectra(torch.linalg.svdvals(jacobian_matrix), title='Jacobian Singular Values', filename=f'spectra/jacobian_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    P_null = get_null_space_projector(jacobian_matrix.T, null_threshold=null_threshold)
    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_kronecker_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth') and try_load:
        checkpoint = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth', map_location=DEVICE)
        A, B = checkpoint['A'],  checkpoint['B']
    else:
        A, B = calculate_kfac_factors_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save({'A': A, 'B': B}, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth')

    # P_null = get_null_space_projector_kron_efficient(B, A, null_threshold=null_threshold)
    P_null = get_null_space_projector(torch.kron(B,A), null_threshold=null_threshold)

    spectra = (torch.linalg.svdvals(B).unsqueeze(1) @ torch.linalg.svdvals(A).unsqueeze(0)).view(-1)
    spectra, _ = torch.sort(spectra, descending=True)
    plot_spectra(spectra, title='Kronecker Singular Values', filename=f'spectra/kronecker_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_kronecker_eigencorrected_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth') and try_load:
        checkpoint = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth', map_location=DEVICE)
        A, B = checkpoint['A'],  checkpoint['B']
    else:
        A, B = calculate_kfac_factors_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save({'A': A, 'B': B}, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac.pth')

    P_null, spectra = get_null_space_projector_kron_corrected(B, A, null_threshold=null_threshold, model=model, loader=approx_loader, criterion=nn.CrossEntropyLoss(), device=DEVICE, layer=layer)
    spectra, _ = torch.sort(spectra, descending=True)
    plot_spectra(spectra, title='Kronecker Singular Values', filename=f'spectra/kronecker_eigencorrected_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_updated_kronecker_optimizer(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load=True):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac_updated.pth') and try_load:
        checkpoint = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac_updated.pth', map_location=DEVICE)
        A, B = checkpoint['A'],  checkpoint['B']
    else:
        A, B = calculate_updated_kfac_factors_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save({'A': A, 'B': B}, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_kfac_updated.pth')

    # P_null = get_null_space_projector_kron_efficient(B, A, null_threshold=null_threshold)
    P_null = get_null_space_projector(torch.kron(B,A), null_threshold=null_threshold)

    spectra = (torch.linalg.svdvals(B).unsqueeze(1) @ torch.linalg.svdvals(A).unsqueeze(0)).view(-1)
    # spectra = (torch.linalg.svdvals(torch.kron(B, A)))
    spectra, _ = torch.sort(spectra, descending=True)
    plot_spectra(spectra, title='Kronecker Singular Values', filename=f'spectra/kronecker_updated_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

def calculate_updated_kronecker_optimizer_no_fac(model, layer_name, lr, approx_loader, train_data_percentage, null_threshold, try_load):
    layer = dict(model.named_modules())[layer_name]
    turn_off_grad_except_layer(model, layer)
    if os.path.exists(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_updated_nofac.pth') and try_load:
        kron = torch.load(f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_updated_nofac.pth', map_location=DEVICE)
    else:
        kron = calculate_updated_kfac_nofac_for_layer(model, approx_loader, nn.CrossEntropyLoss(), DEVICE, layer)
        if try_load:
            torch.save(kron, f'model_cache/exp_approx_{train_data_percentage:.2f}_layer_{layer_name}_updated_nofac.pth')
    P_null = get_null_space_projector(kron, null_threshold=null_threshold)

    spectra = torch.linalg.svdvals(kron)
    spectra, _ = torch.sort(spectra, descending=True)
    plot_spectra(spectra, title='Kronecker Singular Values', filename=f'spectra/kronecker_updated_nofac_spectra_{train_data_percentage:.2f}_{layer_name}.png')

    return ProjectedSGDFlatten(
        [
            {'params': layer.parameters(), 'P_null': P_null}
        ],
        lr=lr
    )

method_to_optimizer = {
    'SGD': calculate_sgd_optimizer,
    'AlphaEdit': calculate_alphaedit_optimizer,
    'Jigsaw': calculate_jigsaw_optimizer,
    'Jigsaw_Hessian': calculate_hessian_optimizer,
    'Jigsaw_GN_Hessian': calculate_gauss_newton_optimizer,
    'Jigsaw_Jacobian': calculate_jacobian_optimizer,
    'Kronecker': calculate_kronecker_optimizer,
    'Kronecker_eigencorrected': calculate_kronecker_eigencorrected_optimizer,
    'Kronecker_updated': calculate_updated_kronecker_optimizer,
    # 'Kronecker_updated_nofac': calculate_updated_kronecker_optimizer_no_fac,
}

def exp(method, null_threshold, train_data_percentage, lr_ft):
    train_loader_pre, test_loader_pre, approx_loader = get_pretrain_and_approx_data(train_data_percentage)
    pretrain_model = load_pretrained_model(train_loader_pre, test_loader_pre, nn.CrossEntropyLoss())

    train_loader_ft, test_loader_ft = get_ft_data()
    optimizer = method_to_optimizer[method](
        pretrain_model,
        "fc2",
        lr_ft,
        approx_loader,
        train_data_percentage,
        null_threshold,
        try_load=True
    )

    # get a partial of method_to_optimizer[method] with only model to fill out later
    fn_get_optimizer_with_model = partial(method_to_optimizer[method], layer_name="fc2", lr=lr_ft, approx_loader=approx_loader, train_data_percentage=train_data_percentage, null_threshold=null_threshold, try_load=False)
    print(f"Starting fine-tuning with method: {method}, Null Threshold: {null_threshold}, TrainData%: {train_data_percentage}")
    ft_accs, ft_losses, pre_accs, pre_losses, weight_differences = fine_tune(
        pretrain_model,
        optimizer,
        train_loader_ft,
        test_loader_ft,
        test_loader_pre,
        nn.CrossEntropyLoss(),
        fn_get_optimizer_with_model,
        "fc2"
    )

    return {
        'ft_accs': ft_accs,
        'ft_losses': ft_losses,
        'pre_accs': pre_accs,
        'pre_losses': pre_losses,
        'weight_differences': weight_differences,
        'method': method,
        'null_threshold': null_threshold,
        'train_data_percentage': train_data_percentage,
        'lr_ft': lr_ft
    }


    
if __name__ == "__main__":
    # do a log linspace between 1e-1 and 1e-11 with 20 points
    vals = torch.linspace(-4, -0.05, steps=16)
    vals = (1 - 10 ** vals).tolist()

    threshold_grids = {
        'Kronecker': vals,
        'Kronecker_eigencorrected': vals,
        'Kronecker_updated': vals,
        'Jigsaw_GN_Hessian': vals,
        'Jigsaw_Hessian': vals,
        'AlphaEdit': vals,
    }
    train_data_perc = 0.06

    data = []
    for method, null_thresholds in threshold_grids.items():
        for null_threshold in null_thresholds:
                print(f"Running experiment: Method={method}, Rank%={null_threshold}, TrainData%={train_data_perc}")
                result = exp(method, null_threshold, train_data_perc, LR_FT)
                data.append(result)
                torch.save({'all_data': data}, f'model_cache/fine_tuning_experiment_results_fc2_recalculation_sweep_{train_data_perc}.pth')
