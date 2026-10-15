import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# eps, dtype = torch.finfo(torch.float32).eps, torch.float32
eps, dtype= torch.finfo(torch.float64).eps, torch.float64


class QuantumGeneticAlgorithm:
    def __init__(self, fun, n_dim, lb, ub, precision=1e-4, pop_size=50, theta=0.1*np.pi,
                 max_iter=100, mutation_rate=0.01, device='cpu', fun_known=None):
        self.fun = fun
        self.fun_known = fun_known
        self.device = device
        self.n_dim = n_dim
        self.theta = torch.tensor(theta, dtype=dtype, device=device)
        self.lb = torch.tensor(lb, dtype=dtype, device=device)
        self.ub = torch.tensor(ub, dtype=dtype, device=device)
        self.precision = torch.tensor(precision, dtype=dtype, device=device)
        self.total_pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate

        self.gene_length, self.chrom_length = self._compute_lengths()
        self._init_population()

    def _init_population(self):
        self.qubit_pop = torch.full((self.total_pop_size, self.chrom_length, 2),
                                    1 / np.sqrt(2), device=self.device)
        tmp = torch.randint(0,2,(self.total_pop_size, self.chrom_length, 2),device=self.device)*2-1
        self.qubit_pop *= tmp

        self.best_solution = None
        self.best_chrom = None
        self.best_fitness = torch.full((1,), float('inf'), device=self.device)
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_chrom_history = []

    def _compute_lengths(self):
        precision = self.precision * torch.ones(self.n_dim, device=self.device)
        gene_length = torch.ceil(torch.log2((self.ub - self.lb) / precision)).int()
        chrom_length = torch.sum(gene_length).item()
        return gene_length, chrom_length

    def observe(self):
        prob_1 = self.qubit_pop[:, :, 1] ** 2
        return (torch.rand_like(prob_1) < prob_1).int()

    def decode(self, chrom_pop):
        decoded = []
        start = 0
        for i, length in enumerate(self.gene_length):
            end = start + length
            segment = chrom_pop[:, start:end]
            gray = torch.cumsum(segment, dim=1) % 2
            weights = torch.pow(0.5, torch.arange(1, length + 1, device=self.device))
            value = ((gray * weights).sum(dim=1) / weights.sum()).view(-1, 1)
            decoded.append(value)
            start = end
        decoded = torch.cat(decoded, dim=1)
        return decoded * (self.ub - self.lb) + self.lb

    def evaluate(self, solution):
        with torch.no_grad():
            return self.fun(solution, self.fun_known).to(self.device)

    def mutation(self):
        mask = torch.rand((self.total_pop_size, self.chrom_length), device=self.device) < self.mutation_rate
        idx = mask.nonzero(as_tuple=True)
        selected = self.qubit_pop[idx[0], idx[1]]
        self.qubit_pop[idx[0], idx[1]] = selected.flip(-1)

        # flip = torch.tensor([[0., 1.], [1., 0.]], device=self.device)
        # flipped = torch.einsum('ij,slj->sli', flip, self.qubit_pop)
        # self.qubit_pop[mask] = flipped[mask]

    def evolve(self):
        for iteration in range(self.max_iter):
            chrom = self.observe()
            solution = self.decode(chrom)
            fitness = self.evaluate(solution)

            best_idx = torch.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx].clone()
                self.best_chrom = chrom[best_idx:best_idx+1].clone()
                self.best_solution = solution[best_idx].clone()

            self.best_fitness_history.append(self.best_fitness.item())
            self.mean_fitness_history.append(torch.mean(fitness).item())

            self._update_qubits(chrom, fitness, self.best_chrom, self.best_fitness)
            if self.mutation_rate > 0:
                self.mutation()

            print(f'{iteration}:  Best: {self.best_fitness.item():.6f}, ' +
                  f'Mean: {self.mean_fitness_history[iteration]:.6f}')

        # self.plot_fitness_history()

        return self.best_solution.cpu().numpy(), self.best_fitness.item()

    def _update_qubits(self, chrom, fitness, best_chrom, best_fitness):

        sign = torch.zeros((self.total_pop_size, self.chrom_length), dtype=dtype, device=self.device)
        sign_rand = 2 * torch.randint(0, 2, (self.total_pop_size, self.chrom_length), dtype=dtype, device=self.device) - 1

        # 条件准备
        is_better = fitness < best_fitness  # 求最小值
        x_xor_best = torch.logical_xor(chrom, best_chrom)
        match_fitness = (best_chrom == is_better.int())

        alpha, beta = self.qubit_pop[:, :, 0], self.qubit_pop[:, :, 1]
        alpha_beta_product = alpha * beta

        greater_zero = alpha_beta_product > eps
        less_zero = alpha_beta_product < -eps
        alpha_zero = torch.abs(alpha) < eps
        beta_zero = torch.abs(beta) < eps

        condition1 = x_xor_best & match_fitness & greater_zero
        sign[condition1] = -1
        condition2 = x_xor_best & match_fitness & less_zero
        sign[condition2] = 1
        condition3 = x_xor_best & (~match_fitness) & greater_zero
        sign[condition3] = 1
        condition4 = x_xor_best & (~match_fitness) & less_zero
        sign[condition4] = -1

        # condition = (x_xor_best * match_fitness * alpha_zero) + (x_xor_best * ~match_fitness * beta_zero)
        condition = x_xor_best & (
            (match_fitness & alpha_zero) | ((~match_fitness) & beta_zero)
        )
        sign[condition] = sign_rand[condition]

        theta = self.theta * sign
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        new_alpha = cos_theta * alpha - sin_theta * beta
        new_beta = sin_theta * alpha + cos_theta * beta

        self.qubit_pop[:, :, 0] = new_alpha
        self.qubit_pop[:, :, 1] = new_beta

    def plot_fitness_history(self):
        if self.best_fitness_history and self.mean_fitness_history:
            plt.figure()
            plt.plot(self.best_fitness_history, label='Best', color='red', alpha=0.5)
            plt.plot(self.mean_fitness_history, label='Mean', color='black', alpha=0.5)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'QEA' + f'Best: {self.best_fitness_history[-1]:.4f}, ' +
                      f'Mean: {self.mean_fitness_history[-1]:.4f}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


class SelectionQA(QuantumGeneticAlgorithm):
    def __init__(self, *args, selection_rate=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.tourn_size = round(self.total_pop_size * selection_rate)

    def evolve(self):
        for iteration in range(self.max_iter):
            chrom = self.observe()
            solution = self.decode(chrom)
            # start_time = time.time()
            fitness = self.evaluate(solution)
            # print(f'fitness time cost: {time.time()-start_time}')

            best_idx = torch.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx].clone()
                self.best_chrom = chrom[best_idx:best_idx+1].clone()
                self.best_solution = solution[best_idx].clone()

            self.best_fitness_history.append(self.best_fitness.item())
            self.mean_fitness_history.append(torch.mean(fitness).item())

            best_chrom, best_fitness = self.selection(chrom, fitness)
            self._update_qubits(chrom, fitness, best_chrom, best_fitness)
            if self.mutation_rate > 0:
                self.mutation()

            # print(f'{iteration}:  Best: {self.best_fitness.item():.6f}, ' +
            #       f'Mean: {self.mean_fitness_history[iteration]:.6f}')

        # self.plot_fitness_history()

        return self.best_solution.cpu().numpy(), self.best_fitness.item()
        # return self.best_solution, self.best_fitness.item()


    def selection(self, chrom, fitness):
        if self.tourn_size != self.total_pop_size:
            base = torch.arange(self.total_pop_size).repeat(self.total_pop_size, 1).to(self.device)
            rand = torch.rand(self.total_pop_size, self.total_pop_size).to(self.device)
            _, indices = torch.sort(rand, dim=1)
            candidates_idx = torch.gather(base, 1, indices)[:, :self.tourn_size]

            # candidates_idx = torch.stack([torch.randperm(self.total_pop_size)[:self.tourn_size] for _ in range(self.total_pop_size)]).to(self.device)

            candidates_values = fitness[candidates_idx]
            winner = torch.argmin(candidates_values, dim=1)
            selected_indices = candidates_idx[torch.arange(self.total_pop_size), winner.view(-1)]
            best_chrom = chrom[selected_indices]
            best_fitness = fitness[selected_indices]
            return best_chrom, best_fitness
        else:
            return self.best_chrom, self.best_fitness


class RankSelectionQA(SelectionQA):
    """SelectionQA with deterministic fitness-rank subsets (minimization).

    The constructor, evolve() return value, and quantum update are inherited.
    N = round(pop_size * selection_rate), as in SelectionQA.
    For zero-based rank r, use N consecutive ranks starting at
    max(0, r - N + 1). The subset includes the current individual;
    at the top boundary it is shifted to ranks 0..N-1.
    Ties are resolved by the original population index (stable sorting).

    N == pop_size preserves SelectionQA's historical-global-best branch.
    Only subset selection is deterministic. Initialization, observation,
    exceptional rotation signs, and mutation retain their original behavior.
    """

    def __init__(self, *args, selection_rate=0.6, **kwargs):
        super().__init__(*args, selection_rate=selection_rate, **kwargs)
        if not 1 <= self.tourn_size <= self.total_pop_size:
            raise ValueError(
                'round(pop_size * selection_rate) must be between 1 and pop_size; '
                f'got {self.tourn_size} for pop_size={self.total_pop_size}.'
            )

    def selection(self, chrom, fitness):
        if self.tourn_size == self.total_pop_size:
            # evolve() records the historical best before calling selection().
            return self.best_chrom, self.best_fitness

        if chrom.shape[0] != self.total_pop_size:
            raise ValueError('chrom must have pop_size rows.')
        if fitness.ndim not in (1, 2) or (
            fitness.ndim == 2 and fitness.shape[1] != 1
        ) or fitness.numel() != self.total_pop_size:
            raise ValueError('fitness must have shape (pop_size,) or (pop_size, 1).')

        values = fitness.reshape(-1)
        # Equal objective values retain population-index order.
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(self.total_pop_size, device=order.device)
        start_ranks = (ranks - self.tourn_size + 1).clamp(min=0)

        # Each window is sorted; its first member is its best. No need to
        # materialize all pop_size * tourn_size candidate indices.
        selected_indices = order[start_ranks]
        return chrom[selected_indices], fitness[selected_indices]


