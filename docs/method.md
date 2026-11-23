# Method

Let $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$ be observed states with a row-stochastic transition matrix $P \in \mathbb{R}^{N\times N}$, and let $p_0$ and $p_1$ be source and target populations. Time $t \in [0, 1]$ is normalized interpolation time, which need not correspond to physical time when the populations are defined from a single snapshot.

## Transition moments

At each observed state

$$m(x_i) = \sum_j P_{ij}\,(x_j - x_i), \qquad C(x_i) = \sum_j P_{ij}\,(x_j - x_i)(x_j - x_i)^\top .$$

$m$ captures the net preferred displacement and $C$ the directions with high transition probability. Both are extended to $\mathbb{R}^d$ by normalized kernel smoothing, which preserves positive semidefiniteness, and $C_\rho = C + \rho I$.

## Randers metric

The regularized Freidlin-Wentzell action gives the $P$-derived Randers metric

$$F_P(x, v) = \sqrt{v^\top C_\rho^{-1} v}\,\sqrt{m^\top C_\rho^{-1} m + \lambda^2} \;-\; v^\top C_\rho^{-1} m ,$$

whose terms are the anisotropic accessibility, the first-moment calibration and the directional alignment; $\lambda > 0$ ensures Randers admissibility. $\rho$ and $\lambda$ are set as multiples of the measured scales $\rho^* = \operatorname{mean}_i \operatorname{tr} C(x_i)/d$ and $\bar a = \operatorname{mean}_i \lVert m(x_i) \rVert_{C_\rho^{-1}}$.

## Geodesic interpolant

$$x_{t,\eta}(x_0, x_1) = (1-t)\,x_0 + t\,x_1 + t(1-t)\,\varphi_\eta(t, x_0, x_1)$$

is trained on the Finsler energy $\mathbb{E}_{(x_0, x_1) \sim p_0 \otimes p_1,\, t}\, F_P(x_{t,\eta}, \dot x_{t,\eta})^2$; energy minimizers are constant-speed geodesics.

## Finsler optimal transport

The learned costs $c_{\eta^*}(x_0, x_1) = \int_0^1 F_P(x_{t,\eta^*}, \dot x_{t,\eta^*})^2\,dt$ define the entropy-regularized coupling $\pi^*_\varepsilon$, solved with Sinkhorn iterations.

## Flow matching

Drawing $(x_0, x_1) \sim \pi^*_\varepsilon$ and transporting each pair along its geodesic gives the probability path $p_t = (x_{t,\eta^*})_\# \pi^*_\varepsilon$. A network $v_\theta$ regresses the interpolant velocity $\dot\mu_t = \dot x_{t,\eta^*}$, and deterministic FFM integrates $dX_t = v_\theta(t, X_t)\,dt$ from $X_0 \sim p_0$.

## Score and flow matching

The stochastic model approximates each conditional bridge by an anisotropic Gaussian path $p_t(x \mid x_0, x_1) = \mathcal{N}\big(x;\, \mu_t,\, \sigma^2 t(1-t)\, C_\rho(\mu_t)\big)$, which injects noise along accessible directions. A drift $\tilde v_\theta$ and a score $s_\psi$ are distilled from these paths by simulation-free conditional flow and score matching, and inference integrates

$$dX_t = \Big[\tilde v_\theta + \tfrac12 g(t)^2 C_\rho(X_t)\, s_\psi + \tfrac12 g(t)^2 \operatorname{div} C_\rho(X_t)\Big] dt + g(t)\, C_\rho(X_t)^{1/2}\, dW_t ,$$

where $g(t)$ sets the noise strength and $C_\rho$ its local anisotropic geometry.
