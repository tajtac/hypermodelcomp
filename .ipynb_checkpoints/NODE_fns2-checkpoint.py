import jax.numpy as jnp
import jax
from jax import grad, vmap, jit, partial, random, jacrev
from jax.experimental.ode import odeint
from jax.experimental import optimizers
from jax.scipy.optimize import minimize

@jit
def forward_pass(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y

@jit
def NN(y0, params):
    f = lambda y, t: forward_pass(jnp.array([y]),params) # fake time argument for ODEint
    return odeint(f, y0, jnp.array([0.0,1.0]))[-1] # integrate between 0 and 1 and return the results at 1

@jit
def NODE_sigma(F, params):
    C = jnp.dot(F.T, F)
    S = NODE_S(C, params)
    return jnp.einsum('ij,jk,kl->il', F, S, F.T)
NODE_sigma_vmap = vmap(NODE_sigma, in_axes=(0, None), out_axes=0)

@jit
def NODE_S(C, params):
    Psi1_params, Psi2_params, Psi4v_params, Psi4w_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params, J_weights, theta = params
    w1, w2, w3, w4, w5, w6 = jnp.abs(J_weights)
    v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)
    I1 = jnp.trace(C)
    C2 = jnp.einsum('ij,jk->ik', C, C)
    I2 = 0.5*(I1**2 - jnp.trace(C2))
    I4v = jnp.einsum('ij,ij',C,V0)
    I4w = jnp.einsum('ij,ij',C,W0)
    
#     I4v = jnp.max([I4v,0.0])
#     I4w = jnp.max([I4w,0.0])
    Cinv = jnp.linalg.inv(C)

    J1 = I1+I2-6
    J2 = I1+I4v-4
    J3 = I1+I4w-4
    J4 = I2+I4v-4
    J5 = I2+I4w-4
    J6 = I4v+I4w-2

    dPsidI1  = NN(I1-3,  Psi1_params ) + w1*NN(J1, J1_params) + w2*NN(J2, J2_params) + w3*NN(J3, J3_params)
    dPsidI2  = NN(I2-3,  Psi2_params ) + w1*NN(J1, J1_params) + w4*NN(J4, J4_params) + w5*NN(J5, J5_params)
    dPsidI4v = NN(I4v-1, Psi4v_params) + w2*NN(J2, J2_params) + w4*NN(J4, J4_params) + w6*NN(J6, J6_params)
    dPsidI4w = NN(I4w-1, Psi4w_params) + w3*NN(J3, J3_params) + w5*NN(J5, J5_params) + w6*NN(J6, J6_params)

#     #Kill dPsidI4v when I4v<1
    dPsidI4v = jnp.max([(I4v-1)*jnp.abs(dPsidI4v), 0])
    dPsidI4w = jnp.max([(I4w-1)*jnp.abs(dPsidI4w), 0])
    
    p = -C[2,2]*(2*dPsidI1 + 2*dPsidI2*(I1 - C[2,2]))
    S = 2*dPsidI1*jnp.eye(3) + 2*dPsidI2*(I1*jnp.eye(3)-C) + 2*dPsidI4v*V0 + 2*dPsidI4w*W0 + p*Cinv
    return S
NODE_S_vmap = vmap(NODE_S, in_axes=0, out_axes=0)