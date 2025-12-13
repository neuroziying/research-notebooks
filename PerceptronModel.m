
eta = 0.01;
max_epochs = 500;
cycles = 50;

% %% prob 2.1 
% 
% for N = 3:30
%     for P = 1:N
%         convergerate = 0;
%         total_epochs = 0;
% 
%         for k = 1:cycles
%             x = zeros(N,P);
%             for i = 1:P
%                 x(:,i) = 10*rand(N,1);
%             end
% 
%             y0 = rand(P,1);
%             for i = 1:P
%                 if y0(i,1)>0.5
%                     y0(i,1) = 1;
%                 else
%                     y0(i,1) = -1;
%                 end
%             end
% 
% 
%             [w,converged,epochs] = perceptron(x,y0,eta,max_epochs);
% 
%             if converged
%                 convergerate = convergerate + 1;
%                 total_epochs = total_epochs + epochs;
%             end
%         end
%         if convergerate > 0
%             mean_epochs = total_epochs / convergerate;
%         else
%             mean_epochs = NaN;
%         end
%         convergerate = (convergerate / cycles) *100;
%         fprintf("\n N=%d,P=%d,mean epochs = %.1f, epoch rate = %.1f",N,P,mean_epochs,convergerate);
%     end
% end
% 
% 
% %% prob 2.2
% 
% n = [10,20,100];
% for n_idx = 1:length(n)
%     N = n(n_idx);
%     P = 10;
%     convergerate = 0;
%     total_epochs = 0;
% 
%     for k = 1:cycles
%         x = zeros(N,P);
%         for i = 1:P
%             x(:,i) = 10*rand(N,1);
%         end
% 
%         y0 = rand(P,1);
%         for i = 1:P
%             if y0(i,1)>0.5
%                 y0(i,1) = 1;
%             else
%                 y0(i,1) = -1;
%             end
%         end
% 
% 
%         [w,converged,epochs] = perceptron(x,y0,eta,max_epochs);
% 
%         if converged
%             convergerate = convergerate + 1;
%             total_epochs = total_epochs + epochs;
%         end
%     end
%     if convergerate > 0
%         mean_epochs = total_epochs / convergerate;
%     else
%         mean_epochs = NaN;
%     end
%     convergerate = (convergerate / cycles) *100;
%     fprintf("\n N=%d,P=%d,mean epochs = %.1f, epoch rate = %.1f",N,P,mean_epochs,convergerate);
% 
% end
% 
% %% prob 2.3 
% 
% alpha_values = 0:0.5:3; 
% convergence_rates = zeros(length(alpha_values), 1);
% mean_epochs_list = zeros(length(alpha_values), 1);
% 
% for N = 3:30
%     for alpha_idx = 1:length(alpha_values)
%         alpha = alpha_values(alpha_idx);
%         P = round(N * alpha);
% 
%         if P < 1
%             P = 1;
%         end
% 
%         converged_count = 0;
%         total_epochs = 0;
% 
%         for k = 1:cycles
% 
%             x = 10 * rand(N, P);
% 
%             y0 = sign(rand(P, 1) - 0.5);
%             y0(y0 == 0) = 1;
% 
%             [w, converged, epochs] = perceptron(x, y0, eta, max_epochs);
% 
%             if converged
%                 converged_count = converged_count + 1;
%                 total_epochs = total_epochs + epochs;
%             end
%         end
% 
%         convergence_rates(alpha_idx) = convergence_rates(alpha_idx) + (converged_count / cycles);
% 
%         if converged_count > 0
%             mean_epochs_list(alpha_idx) = mean_epochs_list(alpha_idx) + (total_epochs / converged_count);
%         end
%     end
% end
% 
% convergence_rates = (convergence_rates / length(3:30)) * 100; 
% mean_epochs_list = mean_epochs_list / length(3:30);
% 
% 
% figure;
% plot(alpha_values, convergence_rates, 'o-', 'LineWidth', 2);
% xlabel('P/N ratio (α)');
% ylabel('Convergence Rate (%)');
% title('convergence probability');
% grid on;

% %% prob 2.4
% 
% N_values = 3:30;
% max_N = max(N_values);
% 
% margin_matrix = zeros(length(N_values), max_N);  
% convergence_matrix = zeros(length(N_values), max_N);  
% margin_histograms = cell(length(N_values), max_N);   
% countN = 0;
% 
% for N = N_values
%     countN = countN + 1;
% 
%     for P = 1:N
%         converged_count = 0;
%         total_epochs = 0;
%         all_margins = [];
% 
%         for k = 1:cycles
%             x = zeros(N,P);
%             for i = 1:P
%                 x(:,i) = 10*rand(N,1);
%             end
% 
%             y0 = rand(P,1);
%             for i = 1:P
%                 if y0(i,1)>0.5
%                     y0(i,1) = 1;
%                 else
%                     y0(i,1) = -1;
%                 end
%             end
% 
%             [w, converged, epochs] = perceptron(x, y0, eta, max_epochs);
% 
%             if converged
%                 converged_count = converged_count + 1;
%                 total_epochs = total_epochs + epochs;
% 
%                 current_margins = zeros(P, 1);
%                 for i = 1:P
%                     current_margins(i) = (w' * x(:, i)) * y0(i);
%                 end
%                 min_margin = min(current_margins); 
%                 all_margins = [all_margins; min_margin];
%             end
%         end
% 
%         if converged_count > 0
%             margin_matrix(countN, P) = mean(all_margins);
%             convergence_matrix(countN, P) = (converged_count / cycles) * 100;
%             margin_histograms{countN, P} = all_margins;
%         else
%             margin_matrix(countN, P) = NaN;
%             convergence_matrix(countN, P) = 0;
%         end
%     end
% end
% figure;
% 
% alpha_ratios = 0.1:0.1:1.0;
% mean_margins_vs_alpha = zeros(size(alpha_ratios));
% for i = 1:length(alpha_ratios)
%     alpha = alpha_ratios(i);
%     margins_at_alpha = [];
%     for countN = 1:length(N_values)
%         N = N_values(countN);
%         P = round(alpha * N);
%         if P >= 1 && P <= N && ~isnan(margin_matrix(countN, P))
%             margins_at_alpha = [margins_at_alpha; margin_matrix(countN, P)];
%         end
%     end
%     mean_margins_vs_alpha(i) = mean(margins_at_alpha);
% end
% plot(alpha_ratios, mean_margins_vs_alpha, 'o-', 'LineWidth', 2);
% xlabel('P/N ratio (α)');
% ylabel('mean Margin');
% title('Margin vs P/N ratio');
% grid on;        
% 
% figure;
% selected_alpha = 0.5;  
% all_margins_selected = [];
% for countN = 1:length(N_values)
%     N = N_values(countN);
%     P = round(selected_alpha * N);
%     if P >= 1 && P <= N && ~isempty(margin_histograms{countN, P})
%         all_margins_selected = [all_margins_selected; margin_histograms{countN, P}];
%     end
% end
% histogram(all_margins_selected, 20);
% xlabel('Margin');
% ylabel('frequency');
% title(sprintf('Margin (P/N=%.1f)', selected_alpha));
% grid on;
% 
% 
% figure;
% fixed_alpha = 0.3;
% margins_vs_N = [];
% for countN = 1:length(N_values)
%     N = N_values(countN);
%     P = round(fixed_alpha * N);
%     if P >= 1 && P <= N && ~isnan(margin_matrix(countN, P))
%         margins_vs_N(countN) = margin_matrix(countN, P);
%     else
%         margins_vs_N(countN) = NaN;
%     end
% end
% plot(N_values, margins_vs_N, 's-', 'LineWidth', 2);
% xlabel('N');
% ylabel('mean Margin');
% title(sprintf('Margin vs N (P/N=%.1f)', fixed_alpha));
% grid on;
% 
% 
% figure;
% fixed_N = 20;
% countN_fixed = find(N_values == fixed_N, 1);
% if ~isempty(countN_fixed)
%     plot(1:fixed_N, margin_matrix(countN_fixed, 1:fixed_N), 'o-', 'LineWidth', 2);
%     xlabel('P');
%     ylabel('mean Margin');
%     title(sprintf('Margin vs P (N=%d)', fixed_N));
%     grid on;
% end

%% prob 2.5
% N = 100;
% f_values = [0.1, 0.3, 0.5, 0.7, 0.9];
% alpha_values = 0.5:0.2:3.0; 
% capacity_results = zeros(length(f_values), length(alpha_values));
% 
% for f_idx = 1:length(f_values)
%     f = f_values(f_idx);
%     for alpha_idx = 1:length(alpha_values)
%         alpha = alpha_values(alpha_idx);
%         P = round(alpha * N);
% 
%         converged_count = 0;
% 
%         for k = 1:cycles
%             x = zeros(N, P);
%             for i = 1:P
%                 for j = 1:N
%                     if rand() < f
%                         x(j, i) = +1;
%                     else
%                         x(j, i) = -1;
%                     end
%                 end
%             end
%             y0 = zeros(P, 1);
%             for i = 1:P
%                 if rand() < f
%                     y0(i) = +1;
%                 else
%                     y0(i) = -1;
%                 end
%             end
% 
%             [w, converged, epochs] = perceptron(x, y0, eta, max_epochs);
% 
%             if converged
%                 converged_count = converged_count + 1;
%             end
%         end
% 
%         convergence_rate = (converged_count / cycles) * 100;
%         capacity_results(f_idx, alpha_idx) = convergence_rate;
%     end
% end
% 
% figure;
% colors = lines(length(f_values));
% hold on;
% for f_idx = 1:length(f_values)
%     plot(alpha_values, capacity_results(f_idx, :), 'o-', ...
%          'Color', colors(f_idx, :), 'LineWidth', 2, ...
%          'DisplayName', sprintf('f=%.1f', f_values(f_idx)));
% end
% xlabel('P/N ratio (α)');
% ylabel('Convergence Rate (%)');
% legend('show', 'Location', 'northeast');
% grid on;
% 
% 
% capacity_points = zeros(1, length(f_values));
% for f_idx = 1:length(f_values)
%     rates = capacity_results(f_idx, :);
%     below_50 = find(rates < 50, 1);
%     if ~isempty(below_50) && below_50 > 1
%         alpha1 = alpha_values(below_50-1);
%         alpha2 = alpha_values(below_50);
%         rate1 = rates(below_50-1);
%         rate2 = rates(below_50);
%         capacity_points(f_idx) = alpha1 + (50 - rate1) * (alpha2 - alpha1) / (rate2 - rate1);
%     else
%         capacity_points(f_idx) = alpha_values(end);
%     end
% end
% figure;
% plot(f_values, capacity_points, 's-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('f');
% ylabel('α_c');
% grid on;

%% prob 2.6

N = 100;
etalist = [1e-1,1e-2,1e-3,1e-4,1e-5];
f_values = 0.5;
alpha_values = [0.5,1,1.5,2,2.5,3];
capacity_results = zeros(length(etalist), length(alpha_values));
for e_idx = 1:length(etalist)
    e = etalist(e_idx);
    f = f_values;
        for alpha_idx = 1:length(alpha_values)
            alpha = alpha_values(alpha_idx);
            P = round(alpha * N);

            converged_count = 0;

            for k = 1:cycles
                x = zeros(N, P);
                for i = 1:P
                    for j = 1:N
                        if rand() < f
                            x(j, i) = +1;
                        else
                            x(j, i) = -1;
                        end
                    end
                end
                y0 = zeros(P, 1);
                for i = 1:P
                    if rand() < f
                        y0(i) = +1;
                    else
                        y0(i) = -1;
                    end
                end

                [w, converged, epochs] = excitatory_perceptron(x, y0, e, max_epochs);

                if converged
                    converged_count = converged_count + 1;
                end
            end

            convergence_rate = (converged_count / cycles) * 100;
            capacity_results(e_idx,alpha_idx) = convergence_rate;
        end
end


figure;
colors = lines(length(alpha_values));
hold on;

for alpha_idx = 1:length(alpha_values)
    plot(etalist, capacity_results(:, alpha_idx), 'o-', 'Color', colors(alpha_idx, :),'LineWidth',2, 'DisplayName', sprintf('α = %.1f', alpha_values(alpha_idx)));
end
xlabel('Learning Rate (η)');
ylabel('Convergence Rate (%)');
legend('show', 'Location', 'northeast');
grid on;
set(gca, 'XScale', 'log');  
% ==========================================================================
function [w, converged, epochs] = perceptron(x,y0,eta,max_epochs)
    [N,P] = size(x);
    w = zeros(N,1);
    converged = false;
    for epochs = 1: max_epochs
        mistake = 0;
        for i = 1:P
            xi = x(:,i);
            yi = y0(i);
            if w'*xi*yi <=0
                w = w + eta * yi * xi;
                mistake = mistake + 1;
            end
        end
        if mistake == 0
            converged = true;
            return;
        end
    end
end

function [w, converged, epochs] = excitatory_perceptron(x,y0,eta,max_epochs)
    [N,P] = size(x);
    w = zeros(N,1);
    theta = 1 ;
    converged = false;

    for epochs = 1: max_epochs
        mistake = 0;
        for i = 1:P
            xi = x(:,i);
            yi = y0(i);
            yx = w'*xi-theta;

            if yx*yi <=0
                w = w + eta * yi * xi;
                mistake = mistake + 1;
                if w < 0
                    w = 0;
                end
            end
        end
        if mistake == 0
            converged = true;
            return;
        end
    end
end