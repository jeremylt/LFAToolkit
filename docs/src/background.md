# Mathematical Background

Local Fourier Analysis was first used by Brandt [1] to analyze the convergence of multi-level adaptive techniques for solving PDEs discretized with finite differences, but the technique has been adapted for multi-level and multi-grid techniques using finite element discretizations.

## Local Fourier Analysis

Local Fourier Analysis considers the local properties of the descretized system via its Fourier modes and the eigenvalues of the associated symbol matrix.
We will describe the arbitrary degree, one dimensional scalar case and extend it to an arbitrary dimension and number of components finite element problem.

First consider a scalar Toeplitz operator ``L_h`` on an infinite one dimensional uniform grid ``G_h``,
This operator is given by

```math
L_h \mathrel{\hat{=}} \left[ s_\kappa \right]_h \left( \kappa \in V \right)

L_h w_h \left( x \right) = \sum_{\kappa \in V} s_\kappa w_h \left( x + \kappa h \right)
```

where ``V \subset \mathcal{Z}`` is a finite index set, ``s_\kappa \in \mathcal{R}`` are constant coefficients and ``w_h \left( x \right)`` is a ``l^2`` function on ``G_h``.

As ``L_h`` is Toeplitz, it can be diagonalized by the standard Fourier modes ``\varphi \left( \theta, x \right) = e^{\imath \theta x / h}``.

If for all grid functions ``\varphi \left( \theta, x \right)`` we have

```math
L_h \varphi \left( \theta, x \right) = \tilde{L}_h \left( \theta \right) \varphi \left( \theta, x \right)
```

then ``\tilde{L}_h \left( \theta \right) = \sum_{\kappa \in V} s_\kappa e^{\imath \theta \kappa}`` is the **symbol** of ``L_h``.

We can extend this to a ``p \times p`` linear system of operators representing a scalar problem on a ``p`` order finite element

```math
\tilde{L}_h =
\begin{bmatrix}
    \tilde{L}_h^{1, 1}  &&  \cdots  &&  \tilde{L}_h^{1, p}  \\
    \vdots              &&  \vdots  &&  \vdots              \\
    \tilde{L}_h^{p, 1}  &&  \cdots  &&  \tilde{L}_h^{p, p}  \\
\end{bmatrix}
```

where ``\tilde{L}_h^{i, j}`` is given by a scalar Toeplitz operator describing how component ``j`` appears in the equation for component ``i``.

The spectral radius of the symbol of an error propagation operator determines how rapidly a relaxation scheme decreases error at a target frequency for a given parameter value.
In this context, low frequencies are given by ``\theta \in T^{low} = \left[ - \pi / 2, \pi / 2 \right)`` and high frequencies are given by ``\theta \in T^{high} = \left[ - \pi / 2, 3 \pi / 2 \right) \setminus T^{low}``.

## High Order Finite Elements

Consider the specific case of a Topeliz operator representing a scalar PDE in 1D with the weak formulation given by Brown in [2],

```math
\int_{\Omega} v \cdot f_0 \left( u, \nabla u \right) + \nabla v : f_1 \left( u, \nabla u \right) = \int_{\Omega} f v, \forall v \in V
```

for some suitable ``V \subseteq H_0^1 \left( \Omega \right)``.
In this equation, ``\cdot`` represents contraction over fields and ``:`` represents contraction over fields and spatial dimensions, both of which are omitted for the sake of clarity in this initial derivation.
Boundary terms have been omitted, as they are not present on the infinite uniform grid ``G_h``.

Selecting a finite element basis, we can discretize the weak form and produce

```math
A u = b.
```

Using the algebraic representation of PDE operators discussed in [2], the PDE operator ``A`` is of the form

```math
A = P^T A_e P
```

```math
A_e = B^T D B
```

where ``P`` represents the element assembly operator, ``B`` is a basis operator which computes the values and derivatives of the basis functions at the quadrature points, and ``D`` is a block diagonal operator which provides a pointwise application of the bilinear form on the quadrature points, to include quadrature weights and the change in coordinates between the physical and reference space.

With a nodal basis of order ``p``, the nodes on the boundary of the element map to the same Fourier mode due to periodicity, and we can thus compute the symbol matrix as

```math
\tilde{A}_h = Q^T \left( A_e \odot \left[ e^{\imath \left( x_i - x_j \right) \theta / h} \right] \right) Q
```

where ``\odot`` represents pointwise multiplication of the elements, ``h`` is the length of the element, and ``i, j \in \left[ 0, 1, \dots, p \right]``.
``Q`` is a ``p + 1 \times p`` matrix that maps the two equivalent solution nodes to the same Fourier mode.

```math
Q =
\begin{bmatrix}
    I    \\
    e_0  \\
\end{bmatrix} =
\begin{bmatrix}
    1       &&  0       &&  \cdots  &&  0       \\
    0       &&  1       &&  \cdots  &&  0       \\
    \vdots  &&  \vdots  &&  \vdots  &&  \vdots  \\
    0       &&  0       &&  \cdots  &&  1       \\
    1       &&  0       &&  \cdots  &&  0       \\
\end{bmatrix}
```

This same computation of the symbol matrix extends to more complex PDE with multiple components and in higher dimensions.

Multiple components are supported by extending the ``p \times p`` system of Toeplitz operators given above to a ``ncomp \cdot p \times ncomp \cdot p`` system of operators.

Tensor products are used to extend this analysis into higher dimensions.
The basis evaluation operators in higher dimensions are given by

```math
B_{interp2d} = B_{interp} \otimes B_{interp}\\
B_{grad2d} =
\begin{bmatrix}
    B_{grad} \otimes B_{interp}  \\
    B_{interp} \otimes B_{grad}  \\
\end{bmatrix}
```

where ``B_{interp}`` and ``B_{grad}`` represent 1D basis interpolation and gradient operators, respectively.

Similarly, the mapping of solution nodes to Fourier modes in higher dimensions is given by

```math
Q_{2d} = Q \otimes Q
```

and an analogous computation can be done for 3D.

Therefore, the symbol matrix for a PDE with arbitrary dimension, order and number of components is given by


```math
\tilde{A}_h = Q^T \left( A_e \odot \left[ e^{\imath \sum_d \left( \mathbf{x}_i - \mathbf{x}_j \right) \mathbf{\theta} / \mathbf{h}} \right] \right) Q
```

where ``\odot`` represents pointwise multiplication of the elements, ``h`` is the length of the element in each dimension, and ``i, j \in \left[ 0, 1, \dots, p \right]``.
``Q`` is a ``p - 1 \times p`` matrix that maps the two equivalent solution nodes to the same Fourier mode.

## P-Multigrid

Multigrid follows the following algorithm:

1. pre-smooth   : ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

2. restrict     : ``r_c := R_{ftoc} \left( b - A u_i \right)``

3. coarse solve : ``A_c e_c := r_c``

4. prolongate   : ``u_i := u_i + P_{ctof} e_c``

5. post-smooth  : ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

where ``f`` and ``c`` represent the fine and coarse grids, respectively, ``R_{ftoc}`` represents the grid restriction operator, ``P_{ctof}`` represents the grid prolongation operator.

The total multigrid error propagation operator is given by

```math
M_{TMG} = S_f \left( I - P_{ctof} A_c^{-1} R_{ftoc} A_f \right) S_f
```

where ``S_f`` represents the error propagation operator for the smoother on the fine grid.

This algorithm describes both h-multigrid and p-multigrid.
While h-multigrid coarsens the mesh by increasing the size of each element, p-multigrid coarsens the mesh by decreasing the order of each element.

To explore the convergence of multigrid techniques, we need to analyze the symbol of the multigrid error propagation operator.
We build the symbol of the p-multigrid error propagation operator in parts.

### Smoothing Operator

Multigrid techniques require error relaxation techniques.
The error propagation operator for a relaxation technique is given by

```math
S = I - M^{-1} A.
```

In the specific case of Jacobi smoothing, ``M`` is given by ``M = diag \left( A \right)``.

The symbol of the error propagation operator is given by

```math
\tilde{S}_h \left( \omega, \theta \right) = I - \tilde{M}_h^{-1} \left( \omega \right) \tilde{A}_h \left( \theta \right)
```

where ``\omega`` is a relaxation parameter.

Specifically, for Jacobi we have

```math
\tilde{S}_h \left( \omega, \theta \right) = I - \omega \tilde{M}_h^{-1} \tilde{A}_h \left( \theta \right)
```

where ``\omega`` is the weighting factor and ``\tilde{M}_h`` is given by

```math
\tilde{M}_h = Q^T diag \left( A_e \right) Q.
```

If multiple pre or post-smoothing passes are used, we have

```math
\tilde{S}_h \left( \omega, \nu, \theta \right) = \left( I - \omega \tilde{M}_h^{-1} \tilde{A}_h \left( \theta \right) \right)^{\nu}
```

where ``\nu`` is the number of smoothing passes.

User defined smoothers are supported, where the user provides ``M^{-1}`` or a function computing ``M^{-1}`` based upon ``A``, and ``\tilde{M}^{-1}_h`` and ``\tilde{S}_h`` are automatically generated and used inside the multigrid symbol matrix.

### Grid Transfer Operators

We consider grid transfer operators for p-type multigrid.
The finite element operator for prolongation from the lower order coarse grid to the high order fine grid is given by 

```math
P_{ctof} = P_f^T P_e P_c\\
P_e = D_{scale} B_{ctof}
```

where ``B_{ctof}`` is a basis interpolation from the coarse basis to the fine basis, ``P_f`` is the fine grid element assembly operator, ``P_c`` is the coarse grid element assembly operator, and ``D_{scale}`` is a scaling operator to account for node multiplicity across element interfaces.

Restriction from the fine grid to the coarse grid is given by the transpose, ``R_{ftoc} = P_{ctof}^T``.

Thus, the symbol of ``P_{ctof}`` is given by

```math
\tilde{P}_{ctof} \left( \theta \right) = Q_f^T \left( \left( D_{scale} B_{ctof} \right) \odot \left[ e^{\imath \sum_d \left( \mathbf{x}_{i, f} - \mathbf{x}_{j, c} \right) \mathbf{\theta} / \mathbf{h}} \right] \right) Q_c
```

and ``\tilde{R}_{ftoc}`` is given by the analagous computation

```math
\tilde{R}_{ftoc} \left( \theta \right) = Q_c^T \left( \left( D_{scale} B_{ctof} \right)^T \odot \left[ e^{\imath \sum_d \left( \mathbf{x}_{i, c} - \mathbf{x}_{j, f} \right) \mathbf{\theta} / \mathbf{h}} \right] \right) Q_f.
```

### Multigrid Error Propagation Symbol

Combining these elements, the symbol of the error propagation operator for p-multigrid is given by

```math
\tilde{E}_{TMG} \left( \omega, \nu, \theta \right) = \tilde{S}_f \left( \omega, \nu, \theta \right) \left[ I - \tilde{P}_{ctof} \left( \theta \right) \tilde{A}_c^{-1} \left( \theta \right) \tilde{R}_{ftoc} \left( \theta \right) \tilde{A}_f \left( \theta \right) \right] \tilde{S}_f \left( \omega, \nu, \theta \right)
```

where ``\tilde{P}_{ctof}`` and ``\tilde{R}_{ftoc}`` are given above, ``\tilde{S}_f`` is given by the smoothing operator, and ``\tilde{A}_c`` and ``\tilde{A}_f`` are derived from the PDE being analyzed.

This can be extended to multi-level analysis by applying this analysis recursively, keeping in mind that ``\tilde{E}_{TMG}`` is the symbol of the multigrid error propagation operator.
The symbol of the multigrid operator is computed by noting that ``E = I - M A``.
