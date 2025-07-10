
"""
Dedalus script simulating compositionally-driven convection in a stellar core. 
This script demonstrates solving an initial value problem in the ball. 
It can be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_ball.py` script can be used to produce
plots from the saved data. The simulation should take roughly 30 cpu-minutes to run.

The strength of gravity for now is defined by an arbitrary Gaussian-type curve for a
non-constant density profile. 

The original Boussinesq problem was non-dimensionalized using the ball radius and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    nu = (Rayleigh / Prandtl)**(-1/2)

The current script employs the anelastic approximation 


We use stress-free boundary conditions, and maintain a constant flux on the outer
boundary.

For incompressible hydro in the ball, we need one tau term each for the velocity
and temperature. Here we choose to lift them to the original (k=0) basis.

The simulation will run to t=20, about the time for the first convective plumes
to hit the top boundary. After running this initial simulation, you can run the
simulation for an addition 20 time units with the command line option '--restart'.

To run, restart, and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 internally_heated_convection.py
    $ mpiexec -n 4 python3 internally_heated_convection.py --restart
    $ mpiexec -n 4 python3 plot_ball.py slices/*.h5
""" 

import sys
from tkinter import E
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import math 
import matplotlib.pyplot as plt 
import random 


# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# PARAMETERS

#RESOLUTION
#Nphi, Ntheta, Nr = 128, 64, 96 
Nphi, Ntheta, Nr = 64, 32, 48 # Seems to be most adequate for testing
#Nphi, Ntheta, Nr = 32, 16, 24  # will run really quickly but ideally we should not use this one, often gives dodgy result

#DIMENSIONLESS NUMBERS

Rayleigh = 1e7 #1e7 = original value. Thermal diffusivity and viscosity are related to the Prandtl and Rayleigh numbers due to way in which original code was non-dimensionalised under Boussinesq assumtion. Maybe we change this?
Prandtl = 1

dealias = 3/2

#CONSTANTS
pi = math.pi 
g = 1 # Dimensionless gravitational acceleration  
omega_0 = 1 # rotational velociity. actually not used anymore in the code due to our non-dimensionalisation
D = 1e-5  #mobility/diffusivity - this value should actually be be smaller or could be defined instead as a function . will greatly impact on phase separation 
T_target = 1e-3 #5e-3   # tanh 10% 
L = 1e-9 # Interface thickness in the Cahn-Hilliard equation
#kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2) 

#SIMULATION PARAMETERS
stop_sim_time = 10000 #20 + 20*restart
timestepper = d3.SBDF2
max_timestep = 0.05
dtype = np.float64
mesh = None

# BASES
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
sphere = ball.surface

# FIELDS
u = dist.VectorField(coords, name='u',bases=ball)# velocity 
p = dist.Field(name='p', bases=ball)# pressure
T = dist.Field(name='T', bases=ball) # Order parameter in Cahn-Hilliard equation (aka phase/concentration)
E = dist.Field(name='E', bases=ball)
#T['g'] = np.clip(T['g'], -1, 1)
#C = dist.Field(name='C', bases=ball) # chemical potential 

#Tau parameters for imposing additional boundary constraints 
tau_p = dist.Field(name='tau_p')
tau_u = dist.VectorField(coords, name='tau_u', bases=sphere)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
tau_T = dist.Field(name='tau_T', bases=sphere)
tau_T1 = dist.Field(name='tau_T1', bases=sphere)

# SUBSTITUTIONS
phi, theta, r = dist.local_grids(ball)

# UNIT VECTORS
#0 = phi, 1 = theta 2 = r

# Unit radial vector on   
er = dist.VectorField(coords, bases=ball.radial_basis)
er['g'][2] = 1 

#General radial vector
rvec = dist.VectorField(coords, bases=ball.radial_basis)
rvec['g'][2] = r

# Unit vector in z direction z = cos(theta)r - sin(theta) theta
ez = dist.VectorField(coords, bases = ball) 
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta)

# Unit vector in x direction 
ex = dist.VectorField(coords, bases = ball)
ex['g'][0] = -np.sin(phi) 
ex['g'][1] = (np.cos(theta)) * (np.cos(phi)) 
ex['g'][2] = (np.sin(theta)) * (np.cos(phi))

#x component of position vector in spherical polars 
r_x = dist.Field(name = 'r_x' , bases = ball)
r_x['g'] = r*(np.sin(theta)) * (np.cos(phi)) 


strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
shear_stress = d3.angular(d3.radial(strain_rate(r=1), index=1))

lift = lambda A: d3.Lift(A, ball.derivative_basis(2), -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1)
grad_T = d3.grad(T) + rvec*lift(tau_T1)

#GRAVITY FUNCTIONS 
#Gaussian-like profile for non-constant density sphere
gravity = dist.Field(name='gravity',bases=ball.radial_basis)
gravity['g'] = -g*np.exp(-(((r-1)/0.5)**2) ) 

#SPONGE LAYERS FOR VELOCITY
tanhr = dist.Field(name = 'tanhr', bases = ball.radial_basis)
tanhr['g'] = 1 + np.tanh((r-1)/0.1)

#KIND OF SPONGE LAYER FOR CAHN-HILLIARD 
tanhT = dist.Field(name='tanhT', bases = ball.radial_basis)
tanhT.fill_random('g', seed=42, distribution='normal', scale=0.001) # Random noise
tanhT.low_pass_filter(scales=0.5)
tanhT['g'] = (np.tanh((r-0.2)/0.05)-np.tanh((r-0.3)/0.05))*0.7

#OTHER FUNCTIONS
#log(rho) in anelastic approximation derivation
logT = dist.Field(name='logT',bases=ball)
logT['g'] = r**2 
#SYSTEM OF EQUATIONS TO BE SOLVED

# Problem
problem = d3.IVP([p, u, T, tau_p, tau_u, tau_T], namespace=locals())

#Mass Continuity 
problem.add_equation(" div(u) + tau_p = - u@grad(logT)") 
#Equation of Motion
#problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) =  -gravity*T_target*T*er - u@grad(u) + p*grad(logT) + 2*cross(ez,u) + 0*cross(ez, cross(ez, (r_x*ex))) + D*(T**3 - T - (L**2)*lap(T))*grad(T) - (u@er)*er*tanhr") # ") 

#fully coupled 
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = -gravity*T_target*T*er - u@grad(u) + p*grad(logT) + 2*cross(ez,u) + D*(T**3 - T - (L**2)*lap(T))*grad(T) - (u@er)*er*tanhr") 
#problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = -gravity*T_target*T*er - u@grad(u) + p*grad(logT) + 2*cross(ez,u) + D*(T**3 - T - (L**2)*lap(T))*grad(T) - (u@er)*er*tanhr") 

#Cahn-Hilliard Equation Typical Form
#problem.add_equation("dt(T) + lift(tau_T) = -u@grad(T) + D*tanhT*lap(T**3 - T - (L**2) * lap(T)) ")
problem.add_equation("dt(T) + lift(tau_T) = -u@grad(T) + D*tanhT*lap(T**3 - T - (L**2) * lap(T)) ") 

#BOUNDARY CONDITIONS
#Original and currently working conditions to use with Cahn-Hilliard in unsplit form
problem.add_equation("shear_stress = 0")  # Stress free
problem.add_equation("radial(u(r=1)) = 0")  # No penetration  - dirichlet
problem.add_equation("T(r=1) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge  - don't get rid of this one 


# SOLVER    
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
"""if not restart:
    T.fill_random('g', seed=42, distribution='normal', scale=0.01) # Random noise
    T.low_pass_filter(scales=0.5)
    T['g'] += 1 - r**2 # Add equilibrium state
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s20.h5')
    initial_timestep = 2e-2
    file_handler_mode = 'append'
    """
"""
if not restart:
    # Generate random noise with values in the range -1 to 1
    random_noise = np.random.normal(loc=0, scale=0.5, size=T['g'].shape)  # Scale adjusted for range -1 to 1
    
    # Fill the 'T' field with the generated random noise
    T['g'] = random_noise
    
    random_noise = np.clip(random_noise, -1, 1)

    # Apply a low-pass filter to smooth out the noise (if desired)
    T.low_pass_filter(scales=0.5)  # Adjust scales as needed

    #T['g'] += 0 # Add equilibrium state
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s20.h5')
    initial_timestep = 2e-2
    file_handler_mode = 'append'
"""
#in theory where we define T as a phi term between 0 and 1
if not restart:
    T.fill_random('g', seed=42, distribution='normal', loc = 0, scale=0.01) # Random noise
    T.low_pass_filter(scales=0.5) # could make smalller 
    #T['g'] = (1 - np.tanh((r-0.1)/0.01))*0.5
    T['g'] += 0.0

    u.fill_random('g', seed=42, distribution='normal', loc = 0, scale=0.1) # Random noise
    u.low_pass_filter(scales=0.5) # could make smalller 

    
    #T.fill_random('g', seed=42, distribution='normal', scale=0.01) # Random noise
    #T.low_pass_filter(scales=0.5)

    #random_noise = np.random.triangular(-1, 0, 1, size=T['g'].shape)
    #T.low_pass_filter(scales=0.1) # could make smalller 
    #T['g'] = random_noise  * tanhT

    #T['g'] += 1 - r**2 # Add equilibrium state
    #T['g'] += 0 # Add equilibrium state

    #T['g'] = np.clip(T['g'], -1, 1)
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep

    
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s63.h5')
    initial_timestep = 2e-2
    file_handler_mode = 'append'

 

# Analysis
u_er = u @ er
E = 0.5 * T_target * T *(u@u) 
E_2 = 0.5 * T_target * T *(u@u) * logT
slices = solver.evaluator.add_file_handler('slices', sim_dt=10, max_writes=1, mode=file_handler_mode)
slices.add_task(T(phi=0), scales=dealias, name='T(phi=0)')
slices.add_task(T(phi=np.pi), scales=dealias, name='T(phi=pi)')
slices.add_task(T(phi=3/2*np.pi), scales=dealias, name='T(phi=3/2*pi)')
slices.add_task(T(r=1), scales=dealias, name='T(r=1)')

slices.add_task(p(phi=0), scales=dealias, name='p(phi=0)')
slices.add_task(p(phi=np.pi), scales=dealias, name='p(phi=pi)')
slices.add_task(p(phi=3/2*np.pi), scales=dealias, name='p(phi=3/2*pi)')
slices.add_task(p(r=1), scales=dealias, name='p(r=1)')

slices.add_task(u_er(phi=0), scales=dealias, name='u(phi=0)')
slices.add_task(u_er(phi=np.pi), scales=dealias, name='u(phi=pi)')
slices.add_task(u_er(phi=3/2*np.pi), scales=dealias, name='u(phi=3/2*pi)')
slices.add_task(u_er(r=1), scales=dealias, name='u(r=1)')

slices.add_task(E(phi=0), scales=dealias, name='E(phi=0)')
slices.add_task(E(phi=np.pi), scales=dealias, name='E(phi=pi)')
slices.add_task(E(phi=3/2*np.pi), scales=dealias, name='E(phi=3/2*pi)')
slices.add_task(E(r=1), scales=dealias, name='E(r=1)')


checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=10, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Analysis
#slices = solver.evaluator.add_file_handler('slices', sim_dt=0.1, max_writes=1, mode=file_handler_mode)
#slices.add_task(T(phi=0), scales=dealias, name='T(phi=0)')
#slices.add_task(T(phi=np.pi), scales=dealias, name='T(phi=pi)')
#slices.add_task(T(phi=3/2*np.pi), scales=dealias, name='T(phi=3/2*pi)')
#slices.add_task(T(r=1), scales=dealias, name='T(r=1)')
#checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=1, max_writes=1, mode=file_handler_mode)
#checkpoints.add_tasks(solver.state)

# CFL
#CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep)
CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=5e-2)

CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
#flow.add_property(0.5*T*u@u, name='E_vol')
#flow.add_property(T, name = 'T')
# Main loop

# Initialize empty lists to store max_u and iteration values
max_u_values = []
iteration_values = []

try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep() 
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            #max_E = flow.max('E_vol')
            #max_T = flow.max('T')
            max_u_values.append(max_u)
            iteration_values.append(solver.iteration)
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, timestep, max_u)) # add all necessary outputs here i.e max E etc.
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# Create a plot of max_u vs. iteration
plt.figure()
plt.plot(iteration_values, max_u_values)
plt.xlabel('Iteration')
plt.ylabel('Max(u)')
plt.title('Max(u) vs. Iteration')
plt.grid(True)

# Save the plot as a PNG file
plot_filename = 'max_u_vs_iteration.png'
plt.savefig(plot_filename)

# Logging the plot filename
logger.info(f"Plot saved as {plot_filename}")


