import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

# ======================================================================================
# BÖLÜM 1: MODELİN TEMEL BİLEŞENLERİ
# ======================================================================================

def b_spline_basis(x, knots, degree):
    """
    Cox-de Boor rekürensiyon formülünü kullanarak B-spline basis fonksiyonlarını hesaplar.
    """
    x = x.squeeze(-1).to(knots.device)
    basis = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).float()
    basis[:, -1] += (x.unsqueeze(-1) == knots[-1]).float().squeeze(-1)
    
    for d in range(1, degree + 1):
        b_prev_1, b_prev_2 = basis[:, :-1], basis[:, 1:]
        n_new_basis = b_prev_1.shape[1]
        
        knots_i, knots_i_d = knots[0:n_new_basis], knots[d:d+n_new_basis]
        knots_i_1, knots_i_d_1 = knots[1:1+n_new_basis], knots[d+1:d+1+n_new_basis]

        denom1, denom2 = knots_i_d - knots_i, knots_i_d_1 - knots_i_1
        num1, num2 = x.unsqueeze(-1) - knots_i, knots_i_d_1 - x.unsqueeze(-1)
        
        term1, term2 = torch.zeros_like(b_prev_1), torch.zeros_like(b_prev_2)
        
        mask1 = denom1 > 1e-8
        if torch.any(mask1): term1[..., mask1] = (num1[..., mask1] / denom1[mask1]) * b_prev_1[..., mask1]
        
        mask2 = denom2 > 1e-8
        if torch.any(mask2): term2[..., mask2] = (num2[..., mask2] / denom2[mask2]) * b_prev_2[..., mask2]
        
        basis = term1 + term2
    return basis

class SplineActivationModule(nn.Module):
    """
    Öğrenilebilir bir aktivasyon fonksiyonu. Kendini sembolik bir fonksiyona benzetmeye çalışır.
    """
    def __init__(self, grid_size, spline_degree, function_library, function_names, complexity_scores):
        super().__init__()
        self.grid_size, self.spline_degree = grid_size, spline_degree
        self.num_coeffs = self.grid_size + self.spline_degree
        self.spline_coeffs = nn.Parameter(torch.randn(self.num_coeffs))
        self.knot_deltas = nn.Parameter(torch.randn(self.grid_size + 1))
        
        self.lib_size = len(function_library)
        self.base_functions = function_library
        self.function_names = function_names
        self.symbolic_params = nn.Parameter(torch.rand(self.lib_size, 4)) # a,b,c,d
        self.complexity_scores = torch.tensor(complexity_scores, dtype=torch.float32)

    def get_knots(self):
        device = self.knot_deltas.device
        deltas = F.softplus(self.knot_deltas)
        deltas_normalized = deltas / deltas.sum() * 2.0
        knots_interior = -1.0 + torch.cumsum(deltas_normalized, dim=0)
        knots = torch.cat([-torch.ones(1, device=device), knots_interior[:-1], torch.ones(1, device=device)])
        return torch.cat([torch.full((self.spline_degree,), knots[0].item(), device=device), knots, torch.full((self.spline_degree,), knots[-1].item(), device=device)])

    def forward(self, x):
        knots = self.get_knots()
        basis = b_spline_basis(x, knots, self.spline_degree)
        if basis.shape[1] < self.num_coeffs:
            basis = F.pad(basis, (0, self.num_coeffs - basis.shape[1]), 'constant', 0)
        elif basis.shape[1] > self.num_coeffs:
            basis = basis[:, :self.num_coeffs]
        return F.linear(basis, self.spline_coeffs).unsqueeze(-1)

    def symbolic_loss(self):
        sample_x = torch.linspace(-1, 1, 100).view(-1, 1).to(self.spline_coeffs.device)
        spline_output = self(sample_x)
        mse_losses = []
        for i in range(self.lib_size):
            a, b, c, d = self.symbolic_params[i]
            if self.function_names[i] == "0":
                 transformed_f = self.base_functions[i](sample_x)
            else:
                 b, c = b * 2, (c - 0.5) * 2
                 transformed_f = a * self.base_functions[i](b * sample_x + c) + d
            mse_losses.append(F.mse_loss(spline_output, transformed_f))
        
        all_mse_losses = torch.stack(mse_losses)
        selection_probs = F.softmax(-all_mse_losses / 1e-2, dim=0)
        
        similarity_loss = torch.sum(selection_probs * all_mse_losses)
        sparsity_loss = -torch.sum(selection_probs * torch.log(selection_probs + 1e-8))
        expected_complexity = torch.sum(selection_probs * self.complexity_scores.to(selection_probs.device))
        
        return similarity_loss, sparsity_loss, expected_complexity

    def get_best_symbolic_info(self, var_name="x"):
        with torch.no_grad():
            mse_losses = self.symbolic_loss_for_inspection()
            inspection_lambda_comp = 0.1 
            combined_score = mse_losses + inspection_lambda_comp * self.complexity_scores.to(mse_losses.device)
            best_idx = torch.argmin(combined_score).item()
            func_name = self.function_names[best_idx]
            
            if func_name == "0":
                return "0.00"

            a, b, c, d = self.symbolic_params[best_idx]
            b, c = b * 2, (c - 0.5) * 2
            return f"{a.item():.2f}*{func_name}({b.item():.2f}*{var_name} + {c.item():.2f}) + {d.item():.2f}"

    def symbolic_loss_for_inspection(self):
        sample_x = torch.linspace(-1, 1, 100).view(-1, 1).to(self.spline_coeffs.device)
        spline_output = self(sample_x)
        mse_losses = []
        for i in range(self.lib_size):
            a, b, c, d = self.symbolic_params[i]
            if self.function_names[i] == "0":
                transformed_f = self.base_functions[i](sample_x)
            else:
                b, c = b * 2, (c - 0.5) * 2
                transformed_f = a * self.base_functions[i](b * sample_x + c) + d
            mse_losses.append(F.mse_loss(spline_output, transformed_f))
        return torch.stack(mse_losses)

class UnifiedNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_alpha = nn.Parameter(torch.randn(2))
    def forward(self, inputs, gumbel_tau=1.0):
        sum_val = torch.sum(inputs, dim=1, keepdim=True)
        prod_val = torch.prod(torch.clamp(inputs, -5, 5), dim=1, keepdim=True)
        op_weights = F.gumbel_softmax(self.op_alpha, tau=gumbel_tau, hard=False)
        return op_weights[0] * sum_val + op_weights[1] * prod_val
    def symbolic_loss(self):
        op_probs = F.softmax(self.op_alpha, dim=0)
        return -torch.sum(op_probs * torch.log(op_probs + 1e-8))
    def get_operation_symbol(self):
        return "+" if torch.argmax(self.op_alpha) == 0 else "*"

class UnifiedKAN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=2, grid_size=5, spline_degree=3):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        
        self.base_functions = [lambda x: torch.zeros_like(x), lambda x: x, lambda x: x**2, torch.sin, torch.exp]
        self.function_names = ["0", "x", "x^2", "sin", "exp"]
        self.complexity_scores = [0.0, 0.1, 0.2, 1.0, 1.5]

        self.layer1_activations = nn.ModuleList([SplineActivationModule(grid_size, spline_degree, self.base_functions, self.function_names, self.complexity_scores) for _ in range(input_dim * hidden_dim)])
        self.hidden_nodes = nn.ModuleList([UnifiedNode() for _ in range(hidden_dim)])
        self.layer2_activations = nn.ModuleList([SplineActivationModule(grid_size, spline_degree, self.base_functions, self.function_names, self.complexity_scores) for _ in range(hidden_dim)])
        self.base_offset = nn.Parameter(torch.randn(1))

    def forward(self, x, gumbel_tau=1.0):
        hidden_outputs = []
        for h in range(self.hidden_dim):
            node_inputs = [self.layer1_activations[h * self.input_dim + i](x[:, i:i+1]) for i in range(self.input_dim)]
            hidden_outputs.append(self.hidden_nodes[h](torch.cat(node_inputs, dim=1), gumbel_tau))
        hidden_tensor = torch.cat(hidden_outputs, dim=1)
        final_output = sum(self.layer2_activations[h](hidden_tensor[:, h:h+1]) for h in range(self.hidden_dim))
        return final_output + self.base_offset

    def loss(self, output, target, lambda_sim, lambda_spar, lambda_comp):
        task_loss = F.mse_loss(output, target)
        total_sim_loss, total_spar_loss, total_comp_loss = 0.0, 0.0, 0.0
        all_splines = list(self.layer1_activations) + list(self.layer2_activations)
        for act in all_splines:
            sim_loss, spar_loss, comp_loss = act.symbolic_loss()
            total_sim_loss += sim_loss
            total_spar_loss += spar_loss
            total_comp_loss += comp_loss
        avg_sim_loss = total_sim_loss / len(all_splines)
        avg_spar_loss = total_spar_loss / (len(all_splines) + len(self.hidden_nodes))
        avg_comp_loss = total_comp_loss / len(all_splines)
        return task_loss + lambda_sim * avg_sim_loss + lambda_spar * avg_spar_loss + lambda_comp * avg_comp_loss

    def get_formula(self):
        final_formula = f"f(x1, x2) ≈ {self.base_offset.item():.3f}"
        hidden_formulas = []
        for h in range(self.hidden_dim):
            op_symbol = self.hidden_nodes[h].get_operation_symbol()
            terms = [self.layer1_activations[h * self.input_dim + i].get_best_symbolic_info(f"x{i+1}") for i in range(self.input_dim)]
            
            # "0.00" terimlerini formülden temizle
            active_terms = [t for t in terms if t != "0.00"]
            if not active_terms:
                h_formula = "0.00"
            else:
                h_formula = f" {op_symbol} ".join([f"({t})" for t in active_terms])

            hidden_formulas.append(f"({h_formula})")

        for h in range(self.hidden_dim):
            outer_term = self.layer2_activations[h].get_best_symbolic_info(f"h{h+1}")
            # Eğer dış terim bir sabite (0*h+d) veya sıfırsa, bu dalı atla
            if f"0.00*{'h'}{h+1}" in outer_term.replace(" ", "") or outer_term == "0.00":
                continue
            
            full_term = outer_term.replace(f"h{h+1}", hidden_formulas[h])
            final_formula += f" + {full_term}"
        return final_formula

# ======================================================================================
# BÖLÜM 2: FORMÜL ÇIKARICI
# ======================================================================================

class FormulaParser:
    def __init__(self, model: UnifiedKAN):
        self.model = model
        self.torch_functions = {"0": lambda x: torch.zeros_like(x), "x": lambda x: x, "x^2": lambda x: torch.pow(x, 2), "sin": torch.sin, "exp": torch.exp}

    def _parse_spline(self, spline_module: SplineActivationModule):
        with torch.no_grad():
            mse_losses = spline_module.symbolic_loss_for_inspection()
            inspection_lambda_comp = 0.1 
            combined_score = mse_losses + inspection_lambda_comp * spline_module.complexity_scores.to(mse_losses.device)
            best_idx = torch.argmin(combined_score).item()
            func_name = spline_module.function_names[best_idx]
            base_func = self.torch_functions[func_name]
            
            if func_name == "0":
                return lambda x: torch.zeros_like(x)

            a, b, c, d = spline_module.symbolic_params[best_idx]
            b, c = b * 2, (c - 0.5) * 2
            return lambda x: a * base_func(b * x + c) + d

    def get_executable_formula(self):
        l1_funcs = [self._parse_spline(act) for act in self.model.layer1_activations]
        node_ops = [node.get_operation_symbol() for node in self.model.hidden_nodes]
        l2_funcs = [self._parse_spline(act) for act in self.model.layer2_activations]
        offset = self.model.base_offset.item()
        input_dim, hidden_dim = self.model.input_dim, self.model.hidden_dim
        def final_symbolic_function(x):
            hidden_outputs = []
            for h in range(hidden_dim):
                node_inputs = [l1_funcs[h * input_dim + i](x[:, i:i+1]) for i in range(input_dim)]
                if node_ops[h] == '+': hidden_val = torch.sum(torch.cat(node_inputs, dim=1), dim=1, keepdim=True)
                else: hidden_val = torch.prod(torch.cat(node_inputs, dim=1), dim=1, keepdim=True)
                hidden_outputs.append(hidden_val)
            hidden_tensor = torch.cat(hidden_outputs, dim=1)
            final_output = sum(l2_funcs[h](hidden_tensor[:, h:h+1]) for h in range(hidden_dim))
            return final_output + offset
        return final_symbolic_function

# ======================================================================================
# BÖLÜM 3: ANA SÜREÇ (AKILLI EĞİTİM VE DOĞRULAMA)
# ======================================================================================

def generate_unified_dataset(n_samples=3000, splits=(0.7, 0.15, 0.15)):
    x = torch.rand(n_samples, 2) * 2 - 1
    noise = torch.randn(n_samples, 1) * 0.02
    y = torch.sin(math.pi * x[:, 0:1]) * 0.5 + x[:, 1:2]**2 + noise
    true_formula = "f(x,y) = 0.5*sin(pi*x) + y^2"
    n_train = int(n_samples * splits[0])
    n_val = int(n_samples * splits[1])
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_test, y_test = x[n_train+n_val:], y[n_train+n_val:]
    print(f"Veri Seti Boyutları: Eğitim={len(x_train)}, Validasyon={len(x_val)}, Test={len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test, true_formula

def train_with_validation_and_verify(
    epochs=30000, lr=1e-3, 
    initial_lambda_sim=0.01, initial_lambda_spar=0.01, initial_lambda_comp=0.02,
    patience=4, lr_decay_factor=0.5, lambda_increase_factor=2.0
):
    x_train, y_train, x_val, y_val, x_test, y_test, true_formula = generate_unified_dataset()
    model = UnifiedKAN(hidden_dim=2, grid_size=5, spline_degree=3) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    current_lambda_sim, current_lambda_spar, current_lambda_comp = initial_lambda_sim, initial_lambda_spar, initial_lambda_comp
    best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
    lr_adjusted, lambdas_adjusted = False, False

    print("\n--- Akıllı Eğitim Döngüsü Başladı ---")
    print(f"Gerçek Formül: {true_formula}")
    initial_tau, final_tau = 2.0, 0.5

    for epoch in range(epochs):
        model.train()
        tau = initial_tau * (final_tau / initial_tau) ** (epoch / (epochs - 1)) if epochs > 1 else final_tau
        output = model(x_train, gumbel_tau=tau)
        loss = model.loss(output, y_train, current_lambda_sim, current_lambda_spar, current_lambda_comp)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch > 0 and (epoch % 500 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                val_output = model(x_val, gumbel_tau=tau)
                val_loss = F.mse_loss(val_output, y_val)
            
            current_formula = model.get_formula()
            print("-" * 80)
            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.4f} | Validation MSE: {val_loss.item():.6f}")
            print(f"  -> Mevcut Formül Tahmini: {current_formula}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  -> Yeni en iyi validasyon kaybı bulundu. Model kaydedildi.")
            else:
                patience_counter += 1
                print(f"  -> Validasyon kaybı iyileşmedi. Sabır: {patience_counter}/{patience}")

            if patience_counter >= patience:
                if not lr_adjusted:
                    lr_adjusted, patience_counter = True, 0
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = old_lr * lr_decay_factor
                    optimizer.param_groups[0]['lr'] = new_lr
                    print(f"  *** MÜDAHALE (1): Sabır tükendi. Öğrenme oranı {old_lr:.1e}'den {new_lr:.1e}'e düşürüldü. ***")
                elif not lambdas_adjusted:
                    lambdas_adjusted, patience_counter = True, 0
                    current_lambda_spar *= lambda_increase_factor
                    current_lambda_comp *= lambda_increase_factor
                    print(f"  *** MÜDAHALE (2): Hala iyileşme yok. Model basitleşmeye zorlanıyor. ***")
                    print(f"      -> Yeni Lambdalar: spar={current_lambda_spar:.3f}, comp={current_lambda_comp:.3f}")
                else:
                    print("  *** MÜDAHALE (3): Tüm müdahalelere rağmen iyileşme yok. Eğitim erken durduruluyor. ***")
                    break

    print("\n" + "="*80)
    print("--- Eğitim Döngüsü Tamamlandı ---")
    if best_model_state is None:
        best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    print("\nEn iyi validasyon sonucunu veren model durumu yüklendi.")
    print("\n--- Formül Çıkarımı ve Nihai Doğrulama (Test Seti Üzerinde) ---")
    model.eval()
    
    learned_formula_str = model.get_formula()
    print(f"\nGerçek Formül      : {true_formula}")
    print(f"Modelin Bulduğu Formül: {learned_formula_str}")
    
    parser = FormulaParser(model)
    symbolic_function = parser.get_executable_formula()
    with torch.no_grad():
        original_model_output = model(x_test)
        original_model_mse = F.mse_loss(original_model_output, y_test)
        symbolic_output = symbolic_function(x_test)
        symbolic_mse = F.mse_loss(symbolic_output, y_test)
        
        print("\n--- Test Seti Üzerinde Performans Doğrulaması ---")
        print(f"Orijinal KAN Modelinin (En İyi Hali) MSE'si: {original_model_mse.item():.6f}")
        print(f"Çıkarılan Sembolik Formülün MSE'si        : {symbolic_mse.item():.6f}")

if __name__ == '__main__':
    train_with_validation_and_verify()
