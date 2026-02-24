import math
from typing import List, Tuple, Optional

# Numerical constants
EPSILON = 1e-15  # Smaller to handle extreme cases
MIN_SIGMA = 1e-6


def phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def Phi_inv(p: float) -> float:
    """Inverse of standard normal CDF (probit function)."""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    
    # Rational approximation
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ]
    
    p_low = 0.02425
    p_high = 1 - p_low
    
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def v_win(t: float, epsilon: float = 0) -> float:
    """V function for win (truncated Gaussian, one-sided)."""
    denom = Phi(t - epsilon)
    if denom < EPSILON:
        return -t + epsilon
    return phi(t - epsilon) / denom


def w_win(t: float, epsilon: float = 0) -> float:
    """W function for win (variance update, one-sided)."""
    v = v_win(t, epsilon)
    w = v * (v + t - epsilon)
    return max(0, min(1 - EPSILON, w))


def v_draw(t: float, epsilon: float) -> float:
    """V function for draw (truncated Gaussian, two-sided)."""
    if epsilon < EPSILON:
        return 0.0
    
    # Draw region: -epsilon < d < epsilon
    # v = (φ(-ε-t) - φ(ε-t)) / (Φ(ε-t) - Φ(-ε-t))
    a = epsilon - t
    b = -epsilon - t
    
    denom = Phi(a) - Phi(b)
    if denom < EPSILON:
        return 0.0
    
    numer = phi(b) - phi(a)
    return numer / denom


def w_draw(t: float, epsilon: float) -> float:
    """W function for draw (variance update, two-sided)."""
    if epsilon < EPSILON:
        return 0.0
    
    a = epsilon - t
    b = -epsilon - t
    
    denom = Phi(a) - Phi(b)
    if denom < EPSILON:
        return 0.0
    
    v = v_draw(t, epsilon)
    # w = v² + (a·φ(a) - b·φ(b)) / denom
    return v**2 + (a * phi(a) - b * phi(b)) / denom


class Rating:
    """Represents a player's skill as a Gaussian distribution."""
    
    def __init__(self, mu: float = 25.0, sigma: float = 25.0/3):
        self.mu = mu
        self.sigma = max(sigma, MIN_SIGMA)
    
    @property
    def variance(self) -> float:
        return max(self.sigma, MIN_SIGMA) ** 2
    
    @property
    def precision(self) -> float:
        return 1.0 / self.variance
    
    @property
    def precision_mean(self) -> float:
        return self.mu / self.variance
    
    def copy(self):
        return Rating(self.mu, self.sigma)
    
    def conservative_rating(self, factor: float = 3.0) -> float:
        return self.mu - factor * self.sigma
    
    def __repr__(self) -> str:
        return f"Rating(mu={self.mu:.2f}, sigma={self.sigma:.2f})"
    
    def state_dict(self):
        return {"mu": self.mu, "sigma": self.sigma}
    
    def load_state_dict(self, state_dict):
        self.mu = state_dict["mu"]
        self.sigma = state_dict["sigma"]


class TrueSkillSystem:
    """TrueSkill rating environment with draw support."""
    
    def __init__(
        self,
        initial_mu: float = 25.0,
        initial_sigma: float = 25.0/3,
        beta: float = 25.0/6,
        tau: float = 25.0/300,
        draw_probability: float = 0.10
    ):
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
        
        # Compute draw margin from draw probability
        # Reference formula: draw_margin = Φ⁻¹((p+1)/2) * √size * β
        # For 1v1 (size=2): draw_margin = Φ⁻¹((p+1)/2) * √2 * β
        # This is NOT normalized - it's in performance units
        if draw_probability > 0:
            self.epsilon = Phi_inv((draw_probability + 1) / 2) * math.sqrt(2) * beta
        else:
            self.epsilon = 0.0
    
    @classmethod
    def with_actual_draw_probability(
        cls,
        actual_draw_prob: float,
        mu: float = 25.0,
        sigma: float = 25.0/3,
        beta: float = 25.0/6,
        tau: float = 25.0/300
    ) -> 'TrueSkillSystem':
        """
        Create environment where equal players have the specified actual draw probability.
        
        The draw_probability parameter in the standard constructor does NOT equal
        the actual P(draw) for equal players. This method computes the correct
        parameter to achieve the desired actual draw probability.
        
        Args:
            actual_draw_prob: Desired P(draw) for two equal-rated players (0 to 1)
            mu, sigma, beta, tau: Standard TrueSkill parameters
            
        Returns:
            TrueSkillSystem environment configured for the desired draw rate
            
        Example:
            # Create environment where equal players draw 15% of the time
            env = TrueSkillSystem.with_actual_draw_probability(0.15)
        """
        if actual_draw_prob <= 0:
            param = 0.0
        elif actual_draw_prob >= 1:
            param = 1.0
        else:
            # Relationship: Φ⁻¹((actual+1)/2) = Φ⁻¹((param+1)/2) * β / √(σ² + β²)
            r = beta / math.sqrt(sigma**2 + beta**2)
            z_actual = Phi_inv((actual_draw_prob + 1) / 2)
            z_param = z_actual / r
            param = 2 * Phi(z_param) - 1
            param = min(1.0, max(0.0, param))
        
        return cls(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=param)
    
    def create_rating(self) -> Rating:
        return Rating(self.mu, self.sigma)
    
    def _performance_variance(self, rating: Rating) -> float:
        """Performance variance = skill variance + performance noise."""
        return rating.variance + self.beta ** 2
    
    def _performance_to_skill_update(
        self, 
        perf_posterior: Rating, 
        skill_prior: Rating
    ) -> Rating:
        """Convert performance-space posterior to skill-space.
        
        This is the key method that properly handles the likelihood factor
        between skill and performance variables.
        """
        c = self._performance_variance(skill_prior)
        
        # Performance posterior in natural parameters
        pi_perf_post = 1.0 / perf_posterior.variance
        tau_perf_post = perf_posterior.mu / perf_posterior.variance
        
        # Performance prior (from skill prior through likelihood)
        pi_perf_prior = 1.0 / c
        tau_perf_prior = skill_prior.mu / c
        
        # Message from truncation factor (in performance space)
        pi_msg = pi_perf_post - pi_perf_prior
        tau_msg = tau_perf_post - tau_perf_prior
        
        # Convert message through likelihood factor to skill space
        if pi_msg > EPSILON:
            denom = 1.0 + pi_msg * self.beta ** 2
            pi_msg_skill = pi_msg / denom
            tau_msg_skill = tau_msg / denom
        else:
            pi_msg_skill = 0.0
            tau_msg_skill = 0.0
        
        # Combine with skill prior
        pi_skill_post = skill_prior.precision + pi_msg_skill
        tau_skill_post = skill_prior.precision_mean + tau_msg_skill
        
        sigma_new = math.sqrt(1.0 / pi_skill_post)
        mu_new = tau_skill_post / pi_skill_post
        
        return Rating(mu_new, sigma_new)
    
    def _get_draw_margin_normalized(self, c: float) -> float:
        """Get draw margin normalized by c (total std dev of performance diff)."""
        return self.epsilon / c if c > EPSILON else 0.0
    
    def rate_1v1(
        self,
        player1: Rating,
        player2: Rating,
        outcome: str = "win",  # "win", "loss", or "draw"
        apply_dynamics: bool = True
    ) -> Tuple[Rating, Rating]:
        """Update ratings after a 1v1 match."""
        # Apply dynamics
        if apply_dynamics:
            p1 = Rating(player1.mu, math.sqrt(player1.variance + self.tau**2))
            p2 = Rating(player2.mu, math.sqrt(player2.variance + self.tau**2))
        else:
            p1 = player1.copy()
            p2 = player2.copy()
        
        # Performance variances
        c1 = self._performance_variance(p1)
        c2 = self._performance_variance(p2)
        c = math.sqrt(c1 + c2)
        
        # Normalized performance difference
        t = (p1.mu - p2.mu) / c
        
        # Normalized draw margin
        eps = self._get_draw_margin_normalized(c)
        
        # Get v and w based on outcome
        if outcome == "win":
            v = v_win(t, eps)
            w = w_win(t, eps)
            sign1, sign2 = 1, -1
        elif outcome == "loss":
            v = v_win(-t, eps)
            w = w_win(-t, eps)
            sign1, sign2 = -1, 1
        elif outcome == "draw":
            v = v_draw(t, eps)
            w = w_draw(t, eps)
            sign1, sign2 = 1, -1
        else:
            raise ValueError(f"Unknown outcome: {outcome}")
        
        # Update in performance space
        mu1_perf = p1.mu + sign1 * (c1 / c) * v
        mu2_perf = p2.mu + sign2 * (c2 / c) * v
        
        var1_perf = c1 * (1 - (c1 / c**2) * w)
        var2_perf = c2 * (1 - (c2 / c**2) * w)
        
        # Convert back to skill space
        perf1 = Rating(mu1_perf, math.sqrt(max(MIN_SIGMA**2, var1_perf)))
        perf2 = Rating(mu2_perf, math.sqrt(max(MIN_SIGMA**2, var2_perf)))
        
        new_p1 = self._performance_to_skill_update(perf1, p1)
        new_p2 = self._performance_to_skill_update(perf2, p2)
        
        return new_p1, new_p2
    
    def rate_ranking(
        self, 
        ranking: List[Rating],
        ties: Optional[List[int]] = None,
        apply_dynamics: bool = True,
        max_iterations: int = 10,
        min_delta: float = 0.0001
    ) -> List[Rating]:
        """
        Update ratings after observing a complete ranking with possible ties.
        
        Args:
            ranking: List of ratings in order [1st, 2nd, 3rd, ...]
            ties: List of tie group IDs. Players with same ID are tied.
                  Example: [0, 1, 1, 2] means positions 2 and 3 are tied.
            apply_dynamics: Whether to add dynamics variance
            max_iterations: Max EP iterations
            min_delta: Convergence threshold
        """
        n = len(ranking)
        if n < 2:
            return [r.copy() for r in ranking]
        
        # Default: no ties
        if ties is None:
            ties = list(range(n))
        
        if n == 2:
            if ties[0] == ties[1]:
                return list(self.rate_1v1(ranking[0], ranking[1], "draw", apply_dynamics))
            else:
                return list(self.rate_1v1(ranking[0], ranking[1], "win", apply_dynamics))
        
        # Apply dynamics
        if apply_dynamics:
            ratings = [
                Rating(r.mu, math.sqrt(r.variance + self.tau**2))
                for r in ranking
            ]
        else:
            ratings = [r.copy() for r in ranking]
        
        # Performance space priors
        perf_var = [self._performance_variance(r) for r in ratings]
        pi_prior = [1.0 / v for v in perf_var]
        tau_prior = [r.mu / v for r, v in zip(ratings, perf_var)]
        
        # Initialize messages from each constraint to each player
        msg = {}
        for k in range(n - 1):
            msg[k] = {k: (0.0, 0.0), k + 1: (0.0, 0.0)}
        
        # Iterate until convergence
        for iteration in range(max_iterations):
            max_change = 0
            
            for k in range(n - 1):
                i, j = k, k + 1
                is_draw = (ties[i] == ties[j])
                
                # Compute cavity for player i
                pi_cav_i = pi_prior[i]
                tau_cav_i = tau_prior[i]
                for c in range(n - 1):
                    if c != k and i in msg[c]:
                        pi_cav_i += msg[c][i][0]
                        tau_cav_i += msg[c][i][1]
                
                # Compute cavity for player j
                pi_cav_j = pi_prior[j]
                tau_cav_j = tau_prior[j]
                for c in range(n - 1):
                    if c != k and j in msg[c]:
                        pi_cav_j += msg[c][j][0]
                        tau_cav_j += msg[c][j][1]
                
                # Convert to mean/variance
                if pi_cav_i > EPSILON:
                    var_cav_i = 1.0 / pi_cav_i
                    mu_cav_i = tau_cav_i / pi_cav_i
                else:
                    var_cav_i = perf_var[i]
                    mu_cav_i = ratings[i].mu
                
                if pi_cav_j > EPSILON:
                    var_cav_j = 1.0 / pi_cav_j
                    mu_cav_j = tau_cav_j / pi_cav_j
                else:
                    var_cav_j = perf_var[j]
                    mu_cav_j = ratings[j].mu
                
                # Compute update
                c_diff = math.sqrt(var_cav_i + var_cav_j)
                t = (mu_cav_i - mu_cav_j) / c_diff
                eps = self._get_draw_margin_normalized(c_diff)
                
                if is_draw:
                    v = v_draw(t, eps)
                    w = w_draw(t, eps)
                else:
                    v = v_win(t, eps)
                    w = w_win(t, eps)
                
                # Posterior for player i
                mu_post_i = mu_cav_i + (var_cav_i / c_diff) * v
                var_post_i = var_cav_i * (1 - (var_cav_i / c_diff**2) * w)
                
                # Posterior for player j
                mu_post_j = mu_cav_j - (var_cav_j / c_diff) * v
                var_post_j = var_cav_j * (1 - (var_cav_j / c_diff**2) * w)
                
                # Compute new messages
                if var_post_i > EPSILON:
                    pi_post_i = 1.0 / var_post_i
                    tau_post_i = mu_post_i / var_post_i
                    new_pi_msg_i = pi_post_i - pi_cav_i
                    new_tau_msg_i = tau_post_i - tau_cav_i
                else:
                    new_pi_msg_i = 0.0
                    new_tau_msg_i = 0.0
                
                if var_post_j > EPSILON:
                    pi_post_j = 1.0 / var_post_j
                    tau_post_j = mu_post_j / var_post_j
                    new_pi_msg_j = pi_post_j - pi_cav_j
                    new_tau_msg_j = tau_post_j - tau_cav_j
                else:
                    new_pi_msg_j = 0.0
                    new_tau_msg_j = 0.0
                
                # Track convergence
                delta_i = abs(new_pi_msg_i - msg[k][i][0])
                delta_j = abs(new_pi_msg_j - msg[k][j][0])
                max_change = max(max_change, delta_i, delta_j)
                
                # Update messages
                msg[k][i] = (max(0, new_pi_msg_i), new_tau_msg_i)
                msg[k][j] = (max(0, new_pi_msg_j), new_tau_msg_j)
            
            if max_change < min_delta:
                break
        
        # Compute final posteriors
        result = []
        for i in range(n):
            pi = pi_prior[i]
            tau = tau_prior[i]
            for k in range(n - 1):
                if i in msg[k]:
                    pi += msg[k][i][0]
                    tau += msg[k][i][1]
            
            if pi > EPSILON:
                mu_perf = tau / pi
                var_perf = 1.0 / pi
            else:
                mu_perf = ratings[i].mu
                var_perf = perf_var[i]
            
            # Convert to skill space
            perf_posterior = Rating(mu_perf, math.sqrt(max(MIN_SIGMA**2, var_perf)))
            skill_posterior = self._performance_to_skill_update(perf_posterior, ratings[i])
            result.append(skill_posterior)
        
        return result
    
    def win_probability(self, player1: Rating, player2: Rating) -> float:
        """Probability that player1 beats player2 (excluding draws)."""
        mu_d = player1.mu - player2.mu
        var_d = player1.variance + player2.variance + 2 * self.beta**2
        c = math.sqrt(var_d)
        eps = self._get_draw_margin_normalized(c)
        return Phi((mu_d / c) - eps)
    
    def get_win_probability(self, player1: Rating, player2: Rating) -> float:
        """Alias for win_probability."""
        return self.win_probability(player1, player2)
    
    def get_draw_probability(self, player1: Rating, player2: Rating) -> float:
        """Probability of a draw between two players."""
        mu_d = player1.mu - player2.mu
        var_d = player1.variance + player2.variance + 2 * self.beta**2
        c = math.sqrt(var_d)
        t = mu_d / c
        eps = self._get_draw_margin_normalized(c)
        return Phi(eps - t) - Phi(-eps - t)
    
    def get_loss_probability(self, player1: Rating, player2: Rating) -> float:
        """Probability that player1 loses to player2."""
        return 1.0 - self.get_win_probability(player1, player2) - self.get_draw_probability(player1, player2)
    
    def match_quality(self, player1: Rating, player2: Rating) -> float:
        """Compute match quality (0-1, higher = more balanced)."""
        var_sum = player1.variance + player2.variance + 2 * self.beta**2
        mu_diff = player1.mu - player2.mu
        
        exp_term = -0.5 * mu_diff**2 / var_sum
        sqrt_term = math.sqrt(2 * self.beta**2 / var_sum)
        
        return sqrt_term * math.exp(exp_term)


def scores_to_ties(scores: list, tolerance: float = 0.0) -> list:
    """
    Convert a list of scores (sorted highest to lowest) to a ties list.
    
    The ties list assigns group numbers where equal group = tied players.
    
    Args:
        scores: List of scores in descending order (highest first)
        tolerance: Scores within this difference are considered tied (default: 0 = exact match only)
        
    Returns:
        List of tie group indices
        
    Examples:
        >>> scores_to_ties([100, 90, 80])
        [0, 1, 2]  # No ties
        
        >>> scores_to_ties([100, 100, 80])
        [0, 0, 1]  # First two tied for 1st
        
        >>> scores_to_ties([100, 90, 90, 80])
        [0, 1, 1, 2]  # Middle two tied for 2nd
        
        >>> scores_to_ties([100, 100, 100])
        [0, 0, 0]  # All tied
        
        >>> scores_to_ties([100, 99, 80], tolerance=2)
        [0, 0, 1]  # 100 and 99 tied within tolerance
    """
    if not scores:
        return []
    
    ties = [0]
    current_group = 0
    
    for i in range(1, len(scores)):
        if scores[i-1] - scores[i] <= tolerance:
            # Within tolerance of previous score - same group
            ties.append(current_group)
        else:
            # Clear difference - new group
            current_group += 1
            ties.append(current_group)
    
    return ties


def ranking_to_ties(ranking: list) -> list:
    """
    Convert a ranking list (with possible duplicates) to a ties list.
    
    Args:
        ranking: List of ranks (1-indexed or 0-indexed, with ties as equal values)
        
    Returns:
        List of tie group indices
        
    Examples:
        >>> ranking_to_ties([1, 2, 3])
        [0, 1, 2]  # No ties
        
        >>> ranking_to_ties([1, 1, 3])
        [0, 0, 1]  # First two tied for 1st
        
        >>> ranking_to_ties([1, 2, 2, 4])
        [0, 1, 1, 2]  # Middle two tied for 2nd
        
        >>> ranking_to_ties([0, 0, 0])
        [0, 0, 0]  # All tied
    """
    if not ranking:
        return []
    
    # Map each unique rank to a tie group
    unique_ranks = sorted(set(ranking))
    rank_to_group = {r: i for i, r in enumerate(unique_ranks)}
    
    return [rank_to_group[r] for r in ranking]
