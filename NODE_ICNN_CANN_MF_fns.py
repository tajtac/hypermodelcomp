import jax.numpy as jnp
import jax
from jax import grad, vmap, jit, random, jacrev
from functools import partial
from jax.experimental.ode import odeint
from jax.experimental import optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan
from jax.nn import softplus
from jax.config import config
config.update("jax_enable_x64", True)

##------------------------------------##
## Common functions
# parameter initialization
# forward pass with and without biases
# train and step functions
##------------------------------------##

def init_params(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

def init_params_b(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = random.split(key)
    Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  return (Ws, bs)

@jit
def forward_pass(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y

@jit
def forward_pass_b(H, params):
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = jnp.matmul(H, Ws[i]) + bs[i]
    H = jnp.tanh(H)
  Y = np.matmul(H, Ws[-1]) + bs[-1]
  return Y

@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)

def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
    train_loss = []
    val_loss = []
    for it in range(nIter):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch,:], Y[idx_batch,:])   
       # opt_state = step(loss, it, opt_state, X, Y)            
        if it % 100 == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss

##-------------------------------------##
## NODE functions
# integration of the neural network
# compute PK2 using NODEs for derivatives
# prediction of sigma by push fwd
##-------------------------------------##

# NODE integration with odeint from jax
@jit
def NODE_old(y0, params):
    f = lambda y, t: forward_pass(jnp.array([y]),params) # fake time argument for ODEint
    return odeint(f, y0, jnp.array([0.0,1.0]))[-1] # integrate between 0 and 1 and return the results at 1

#The same function as NODE_old except using Euler integration
@jit
def NODE(y0, params, steps = 100):
    t0 = 0.0
    dt = 1.0/steps
    body_func = lambda y,t: (y + forward_pass(jnp.array([y]), params)[0]*dt, None)
    out, _ = scan(body_func, y0, jnp.linspace(0,1,steps), length = steps)
    return out
NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

# PK2 stress prediction using NODE.
# this is the main function for an arbitrary deformation C
@jit
def NODE_S(C, params):
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

    I1 = I1-3
    I2 = I2-3
    Iv = Iv-1
    Iw = Iw-1
    J1 = a[0]*I1+(1-a[0])*I2
    J2 = a[1]*I1+(1-a[1])*Iv
    J3 = a[2]*I1+(1-a[2])*Iw
    J4 = a[3]*I2+(1-a[3])*Iv
    J5 = a[4]*I2+(1-a[4])*Iw
    J6 = a[5]*Iv+(1-a[5])*Iw

    Iv, Iw, J1, J2, J3, J4, J5, J6 = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    Psi1 = NODE(I1,  I1_params)
    Psi2 = NODE(I2,  I2_params)
    Psiv = NODE(Iv,  Iv_params)
    Psiw = NODE(Iw,  Iw_params)
    Phi1 = NODE(J1,  J1_params)
    Phi2 = NODE(J2,  J2_params)
    Phi3 = NODE(J3,  J3_params)
    Phi4 = NODE(J4,  J4_params)
    Phi5 = NODE(J5,  J5_params)
    Phi6 = NODE(J6,  J6_params)
    
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

# Prediction of cauchy stress by computing PK2 and push forward
@jit
def NODE_sigma(F, params):
    C = jnp.dot(F.T, F)
    S = NODE_S(C, params)
    return jnp.einsum('ij,jk,kl->il', F, S, F.T)
NODE_sigma_vmap = vmap(NODE_sigma, in_axes=(0, None), out_axes=0)

# Prediction of stress for particular type of deformation gradient
@jit
def NODE_lm2sigma(lamb, params):
    lamb1 = lamb[0]
    lamb2 = lamb[1]
    lamb3 = 1/(lamb1*lamb2)
    F = jnp.array([[lamb1, 0, 0],
                   [0, lamb2, 0],
                   [0, 0, lamb3]])
    return NODE_sigma(F, params)[[0,1],[0,1]]
NODE_lm2sigma_vmap = vmap(NODE_lm2sigma, in_axes=(0, None), out_axes = 0)


##------------------------------------##
## ICNN functions
# ininialization for icnn
# special forward pass
##------------------------------------##

def init_params_icnn(layers, key):
  Wz = []
  Wy = []
  bs = []

  std_glorot = jnp.sqrt(2/(layers[0] + layers[1]))
  key, subkey = random.split(key)
  Wy.append(random.normal(subkey, (layers[0], layers[1]))*std_glorot)

  for i in range(1,len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = random.split(key)
    Wz.append(-3 + random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    Wy.append(random.normal(subkey, (layers[0], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  return [Wz, Wy, bs]

@jit
def icnn_forwardpass(Y, params):
  Wz, Wy, bs = params
  N_layers = len(Wy)
  Z = softplus(jnp.matmul(Y, Wy[0]) + bs[0])**2
  for i in range(1, N_layers - 1):
    Z = jnp.matmul(Z, softplus(Wz[i-1])) + jnp.matmul(Y, Wy[i]) + bs[i]
    Z = softplus(Z)**2
  Z = jnp.matmul(Z, jnp.exp(Wz[-1])) + jnp.matmul(Y, Wy[-1]) + bs[-1]
  return Z

##----------------------------------------------##
## CANN functions 
# initialize weights
# eval energy for isotropic material
# eval derivatives of energy isotropic material
##----------------------------------------------##

def init_params_cann(key):
    Ws = []
    std_glorot = jnp.sqrt(1./2.)
    for i in range(4):
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (3, 1))*std_glorot)
    return Ws

def CANN_psi(Inorm,Ws):
    # Inorm assume normalized scalar, e.g. = (I1-3)/normalization
    # powers are not fully connected, defined operation
    # so dont see how to do general, rather, 3 different input
    YIp = jnp.array([Inorm,Inorm**2,Inorm**3])
    # for each of the powers, multiply by positive weights
    # then pass through activation functions
    # 1. Identity activation
    # note: element-wise operation
    Yh1Ip = YIp*Ws[0]**2
    # 2. Exponential activation
    # note: element-wise operations
    Yh2Ip = jnp.exp(YIp*Ws[1]**2)-jnp.ones(YIp.shape)
    # multiply by the next set of weights and add to output
    # note: here yes dot products and adding to scalar 
    Z = jnp.dot(Yh1Ip.transpose(),Ws[2]**2)+jnp.dot(Yh2Ip.transpose(),Ws[3]**2)
    return Z

def CANN_dpsidInorm(Inorm,Ws):
    # Inorm assume normalized scalar, e.g. = (I1-3)/normalization
    # powers are not fully connected, defined operation
    # so dont see how to do general, rather, 3 different input
    YIp = jnp.array([Inorm,Inorm**2,Inorm**3])
    dYIpdI = jnp.array([jnp.ones(Inorm.shape),2.*Inorm,3.*Inorm**2])
    # for each of the powers, multiply by positive weights
    # then pass through activation functions, dot and not matmul
    # 1. Identity activation
    # note: element-wise operations
    Yh1Ip = YIp*Ws[0]**2 
    dYh1IpdI = dYIpdI*Ws[0]**2
    # 2. Exponential activation
    # note: element-wise operation
    Yh2Ip = jnp.exp(YIp*Ws[1]**2)-jnp.ones(YIp.shape)
    dYh2IpdI = jnp.exp(YIp*Ws[1]**2)*Ws[1]**2*dYIpdI
    # multiply by the next set of weights and add to output
    Z = jnp.dot(Yh1Ip.transpose(),Ws[2]**2)+jnp.dot(Yh2Ip.transpose(),Ws[3]**2)
    dZdI = jnp.dot(dYh1IpdI.transpose(),Ws[2]**2) + jnp.dot(dYh2IpdI.transpose(),Ws[3]**2)
    return dZdI

