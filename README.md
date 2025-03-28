# Weights for TempSal
[Google drive](https://drive.google.com/drive/folders/160WB1YrPAjNYy787jP1pmffl9Xv0gLw6)

# Basic Problem description

### Inputs

- An image I
- A target region R
- Saliency contrast $\nabla S$ → given by user

### Objective

distinguishes between salient and non-salient patches and pushes for manipulation that matches the saliency contrast ∆S.

### Patches

We define 2 databases of patches of size $w*w$

- $D⁺=\{p : S_i(p)\geq\tau^+\}$, where p is a patch and this database is composed of patches with a high saliency
- $D⁻=\{p : S_i(p)\leq\tau^-\}$, containing patches of low saliency

The two parameters $\tau⁺ and \tau⁻$ are found via optimization

→ Our task is to augment the saliency of patches belonging to R (the masked region) and decrease the one of the patches outside this masked region $\not \in R$

## The optimization

We use the energy function, where J is the manipulated image ! We want to **minimize E →** generate image J with $S_J$ (saliency map of modified image) such that the saliency contrast between R and the rest is the wanted $\Delta S$  

$$
E(J, D⁺,D⁻)=E⁺+ E⁻ + \lambda E^\nabla
$$

Where the elements are

- $E⁺(J, D⁺) = \sum_{q\in R}min_{p\in D⁺}D(q,p)$
- $E⁻(J, D⁻) = \sum_{q\not\in R}min_{p\in D⁻}D(q,p)$

Where q, p are patches and D(q, p) is the sum of squared distances over {L, a, b}  color channels between q and p

- $E^\nabla(J,I)=||\nabla J-\nabla I||²$ : Preserve the gradients of the original image I, this balance is controlled by the parameter $\lambda$

## The algo

The higher the threshold τ⁺ the more salient will be the patches in D+ and in return those in R and in the same way, the lower the threshold τ⁻ the less salient will be the patches
in D− and in return those outside of R

→ **Perform an approximate greedy search** to minize the objective function
