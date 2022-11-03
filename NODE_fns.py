import jax.numpy as jnp
import jax
from jax import grad, vmap, jit, partial, random, jacrev
from jax.experimental.ode import odeint
from jax.experimental import optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan

@jit
def forward_pass(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y

@jit
def NODE_old(y0, params):
    f = lambda y, t: forward_pass(jnp.array([y]),params) # fake time argument for ODEint
    return odeint(f, y0, jnp.array([0.0,1.0]))[-1] # integrate between 0 and 1 and return the results at 1

#The same function as NODE_old except using Euler integration
@jit
def NODE(y0, params, steps = 100):
    t0 = 0.0
    dt = 100.0/steps
    body_func = lambda y, t: (y + forward_pass(jnp.array([y, t]), params)[0]*dt, None)
    out, ys = scan(body_func, y0, jnp.linspace(0,100,steps), length = steps)
    return out, ys
NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def NODE_sigma(F, params, norm):
    C = jnp.dot(F.T, F)
    S = NODE_S(C, params, norm)
    return jnp.einsum('ij,jk,kl->il', F, S, F.T)
NODE_sigma_vmap = vmap(NODE_sigma, in_axes=(0, None, None), out_axes=0)

@jit
def NODE_lm2sigma(lamb, params, norm):
    lamb1 = lamb[0]
    lamb2 = lamb[1]
    lamb3 = 1/(lamb1*lamb2)
    F = jnp.array([[lamb1, 0, 0],
                   [0, lamb2, 0],
                   [0, 0, lamb3]])
    return NODE_sigma(F, params, norm)[[0,1],[0,1]]
NODE_lm2sigma_vmap = vmap(NODE_lm2sigma, in_axes=(0, None, None), out_axes = 0)
 
@jit
def NODE_S(C, params, norm):
    I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params, I_weights, theta, Psi1_bias, Psi2_bias = params
    a = 1/(1+jnp.exp(-I_weights))
    v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)
    I1 = jnp.trace(C)
    C2 = jnp.einsum('ij,jk->ik', C, C)
    I2 = 0.5*(I1**2 - jnp.trace(C2))
    Iv = jnp.einsum('ij,ij',C,V0)
    Iw = jnp.einsum('ij,ij',C,W0)
    Cinv = jnp.linalg.inv(C)

    I1 = (I1-3)
    I2 = (I2-3)
    Iv = Iv-1
    Iw = Iw-1
    J1 = a[0]*I1+(1-a[0])*I2
    J2 = a[1]*I1+(1-a[1])*Iv
    J3 = a[2]*I1+(1-a[2])*Iw
    J4 = a[3]*I2+(1-a[3])*Iv
    J5 = a[4]*I2+(1-a[4])*Iw
    J6 = a[5]*Iv+(1-a[5])*Iw

    I1norm = I1/norm[0]
    I2norm = I2/norm[1]

    # Iv, Iw, J2, J3, J4, J5, J6 = 0.0,0.0,0.0,0.0,0.0,0.0,0.0
    Psi1 = NODE(I1norm,  I1_params)[0]*norm[2]
    Psi2 = NODE(I2norm,  I2_params)[0]*norm[3]
    Psiv = NODE(Iv,  Iv_params)[0]
    Psiw = NODE(Iw,  Iw_params)[0]
    Phi1 = NODE(J1,  J1_params)[0]
    Phi2 = NODE(J2,  J2_params)[0]
    Phi3 = NODE(J3,  J3_params)[0]
    Phi4 = NODE(J4,  J4_params)[0]
    Phi5 = NODE(J5,  J5_params)[0]
    Phi6 = NODE(J6,  J6_params)[0]
    
    Psiv = jnp.max([Psiv, 0])
    Psiw = jnp.max([Psiw, 0])
    Phi1 = jnp.max([Phi1, 0])
    Phi2 = jnp.max([Phi2, 0])
    Phi3 = jnp.max([Phi3, 0])
    Phi4 = jnp.max([Phi4, 0])
    Phi5 = jnp.max([Phi5, 0])
    Phi6 = jnp.max([Phi6, 0])
    
    Psi1 = Psi1 +     a[0]*Phi1 +     a[1]*Phi2 +     a[2]*Phi3 + jnp.exp(Psi1_bias)
    Psi2 = Psi2 + (1-a[0])*Phi1 +     a[3]*Phi4 +     a[4]*Phi5 + jnp.exp(Psi2_bias)
    Psiv = Psiv + (1-a[1])*Phi2 + (1-a[3])*Phi4 +     a[5]*Phi6
    Psiw = Psiw + (1-a[2])*Phi3 + (1-a[4])*Phi5 + (1-a[5])*Phi6

    p = -C[2,2]*(2*Psi1 + 2*Psi2*((I1+3) - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*((I1+3)*jnp.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0
    return S
NODE_S_vmap = vmap(NODE_S, in_axes=0, out_axes=0)

def init_params(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

def node_traj(params, inp):
    out = []
    for i in range(inp.shape[0]):
        out.append(NODE(inp[i][1]), params)
    return out