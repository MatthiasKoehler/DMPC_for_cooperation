"""
Collection of custom methods.
matthias.koehler@ist.uni-stuttgart.de
====
Uses (imports):
    - casadi
    - numpy
"""
import casadi as cas
import numpy as np
          
class agent:
    """A discrete-time system with individual states and inputs, i.e. without dynamic coupling to other agents.
        
    Methods:
    - set_constraints
    
    Attributes:
    - id (str): identifier, usually an index.
    - state_dim
    - input_dim
    - dynamics
    - output_dim
    - output_map
    - current_state
    - current_time
    - neighbours
    - stage_cost
    - state_constraints (dict): Containing linear inequality constraints (A*x <= b with keys 'A' and 'b'), equality constraints (Aeq*x <= b with keys 'Aeq' and 'beq')
    - input_constraints (dict): Containing linear inequality constraints (A*u <= b with keys 'A' and 'b'), equality constraints (Aeq*u <= b with keys 'Aeq' and 'beq')
    """
    
    def __init__(self, id = None, state_dim = 1, input_dim = 1, dynamics = None, output_dim = None, output_map = None, initial_time = 0, initial_state = None, neighbours = None, box_state_constraints=None, box_input_constraints=None, stage_cost=None):
        """
        Initialise an agent.
        
        Keyword arguments:
        - id: identifier, usually an index.
        - state_dim (int): the dimension of the state. (default 1)
        - input_dim (int): the dimension of the input. (default 1)
        - dynamics (casadi Function): vector field of the system. (default None) 
        - output_dim (int): the dimension of the output. (default state_dim)
        - output_map (casadi Function): function from state and input to output. (default output is state)
        - initial_time (int): initial time for internal time-keeping. (default 0)
        - initial_condition (numpy array): initial state at the initial time. (default 0)
        - neighbours (list with agents): neighbours of this agent. (default None)
        - box_state_constraints (array): Contains a lower (first column) and upper bound (second column) for each state variable (rows).
        - box_input_constraints (array): Contains a lower (first column) and upper bound (second column) for each input variable (rows).
        """
        # Set values as provided.
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dynamics = dynamics
        if output_dim is None:
            self.output_dim = state_dim
        else:
            self.output_dim = output_dim
            
        self.id = str(id)
        
        # Define a symbolic state, input and output.
        self._state = cas.SX.sym('x', self.state_dim)
        self._input = cas.SX.sym('u', self.input_dim)
        self._output = cas.SX.sym('y', self.output_dim)
        
        # Define the output map.
        if output_map is None:
            self.output_map = cas.Function('h', [self._state, self._input], [self._state], ['x', 'u'], ['h(x,u)'])
        else:
            self.output_map = output_map
        
        # Set constraints of the agent. If no box constraints are passed, the constraints are initialised to be empty.
        self.set_constraints(box_state_constraints=box_state_constraints, box_input_constraints=box_input_constraints)          
        
        # Set the initial time (internal clock) and the initial state.
        self.current_time = initial_time
        if initial_state is None:
            self.current_state = np.zeros((self.state_dim, 1))
        else:
            self.current_state = initial_state
        
        if neighbours is None:    
            self.neighbours = []
        else:
            self.neighbours = neighbours
            
        self.stage_cost = stage_cost
        
        
    def set_constraints(self, box_state_constraints=None, box_input_constraints=None):
        """
        Set constraints of the agents.
        
        Keyword arguments:
        - box_state_constraints (array): Contains a lower (first column, 'lb') and upper bound (second column, 'ub') for the state vector ('x'), i.e. lb <= x <= ub element-wise.
        - box_input_constraints (array): Contains a lower (first column, 'lb') and upper bound (second column, 'ub') for the input vector ('u'), i.e. lb <= u <= ub element-wise.
        """
        # Define state constraints.
        self.state_constraints = {"A": np.empty, "b": np.empty}
        if box_state_constraints is not None:
            self.state_constraints["A"] = np.vstack((-np.eye(self.state_dim), np.eye(self.state_dim)))
            self.state_constraints["b"] = np.vstack((-box_state_constraints[:,0:1]*np.ones((self.state_dim,1)), box_state_constraints[:,1:2]*np.ones((self.state_dim,1))))

        
        # Define input constraints.
        self.input_constraints = {"A": np.empty, "b": np.empty}
        if box_input_constraints is not None:
            self.input_constraints["A"] = np.vstack((-np.eye(self.input_dim), np.eye(self.input_dim)))
            self.input_constraints["b"] = np.vstack((-box_input_constraints[:,0:1]*np.ones((self.input_dim,1)), box_input_constraints[:,1:2]*np.ones((self.input_dim,1))))

def MPC_for_cooperation(agent, horizon=1, terminal_cost_matrix=None, warm_start=None, coop_outputs_neighbours=None, solver=None):
    """Implementation of the MPC problem that each agents solves. The problem is set up in casadi and solved.
    Returns the objective value, optimal input sequence, cooperative output, and cooperative equilibrium as well as the dictionary containing the result of ipopt.
    
    The explicit relation of the cooperative output and the cooperative equilibrium is not used. Instead, it is defined implicitely by adding a suitable constraint to the optimisation problem.
    
    Arguments:
    - agent (mkmpc object): Current agent.
    - horizon (int): Prediction horizon of the MPC optimisation problem.
    - terminal_cost_matrix (array): Cost matrix that defines the terminal cost and the terminal set as outlined in the paper. If none is provided, a terminal equality constraint is used. (default None)
    - warm_start (list of arrays): Warm start for the MPC problem. If None is provided, zeros is used. The ordering should be:
      sequence of inputs, sequence of states, cooperation input, cooperation state, cooperation output
    - coop_outputs_neighbours (list of arrays): Communicated cooperative outputs of the neighbours. 
      Ordering corresponds to ordering of 'neighbours' attribute of the agent. If None is provided, the cooperation outputs of the neighbours (i.e. of the objects) are taken, which is the intended behaviour. This attribute servers mainly for initialisation or tests. (default None)
    - solver (string): One of casadis solver options. Default: ipopt
    """
    # Initialise needed objects.
    constraints = []
    constraints_lb = []
    constraints_ub = []
    objective = cas.MX(0)
    
    # Shorthands.
    n = agent.state_dim
    q = agent.input_dim
            
    # Create decision variables.
    # Time steps are stacked beneath each other.
    u = cas.MX.sym('u', q*horizon, 1)
    x = cas.MX.sym('x', n*(horizon + 1), 1)
    uc = cas.MX.sym('xc', q, 1)
    xc = cas.MX.sym('xc', n, 1)
    yc = cas.MX.sym('yc', agent.output_dim, 1)
    
    # Set the initial condition as a constraints.
    constraints.append(x[0 : n, 0] - agent.current_state)
    # This constraint is an equality constraint.
    constraints_lb.append(np.zeros((n,1)))
    constraints_ub.append(np.zeros((n,1)))
    
    # Create constraints containing the dynamics.
    for t in range(horizon):
        constraints.append(agent.dynamics(x[t*n : (t+1)*n, 0], u[t*q : (t+1)*q, 0]) - x[(t+1)*n : (t+2)*n, 0])
        # These constraints are equality constraints.
        constraints_lb.append(np.zeros((n,1)))
        constraints_ub.append(np.zeros((n,1)))
    
    # Set the state constraints.
    for t in range(horizon):
        A = agent.state_constraints["A"]
        b = agent.state_constraints["b"]
        constraints.append(A@x[t*n : (t+1)*n, 0])
        constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
        constraints_ub.append(b)
    # Set the input constraints.
    for t in range(horizon):
        A = agent.input_constraints["A"]
        b = agent.input_constraints["b"]
        constraints.append(A@u[t*q : (t+1)*q, 0])
        constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
        constraints_ub.append(b)
        
    # Set the terminal constraint.
    if terminal_cost_matrix is None:
        # If no terminal cost matrix is provided, use a terminal equality constraint.
        constraints.append(agent.dynamics(x[horizon*n : (horizon+1)*n, 0], uc) - xc)
        constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
        constraints_ub.append(np.zeros((n,1)))
    else:
        raise NotImplementedError('A terminal cost is not implemented yet.')
    
    # Add a constraint that links the cooperation equilibrium to the cooperation output.
    constraints.append(agent.output_map(xc, uc) - yc)
    constraints_lb.append(np.zeros((agent.output_dim,1)))  # This constraint is an equality constraint.
    constraints_ub.append(np.zeros((agent.output_dim,1)))
    # Add a constraint that enforces the cooperation equilibrium to be an equilibrium.
    constraints.append(agent.dynamics(xc, uc) - xc)
    constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
    constraints_ub.append(np.zeros((n,1)))
    
    # Add a constraint for the admissible cooperation output set.
    A = agent.cooperation_output_constraint["A"]
    b = agent.cooperation_output_constraint["b"]
    constraints.append(A@yc)
    constraints_lb.append(-np.inf*np.ones((b.shape[0], 1)))
    constraints_ub.append(b)
    
    ## Create the objective function:
    # Sum up the stage cost.
    stage_cost = agent.stage_cost
    for t in range(horizon):
        objective += stage_cost(x[t*n : (t+1)*n, 0], u[t*q : (t+1)*q, 0], xc, uc)
    # Add the cooperative cost.
    for neigh_agent in agent.neighbours:
        objective += agent.bilat_coop_cost(yc, neigh_agent.current_cooperation_output)
        objective += agent.bilat_coop_cost(neigh_agent.current_cooperation_output, yc)

    # Set warm start of solver.
    if warm_start is None:
        # Initialise with zeros if no warm start was provided.
        warm_start = np.zeros((u.shape[0] + x.shape[0] + uc.shape[0] + xc.shape[0] + yc.shape[0], 1))
    
    # Create optimisation object.
    nlp = {'x':cas.vertcat(u, x, uc, xc, yc), 'f':objective, 'g':cas.vertcat(*constraints)}

    solver_options = {}
    if solver is None or solver == 'ipopt':
        solver_options["fixed_variable_treatment"] = 'make_constraint'
        solver_options["print_level"] = 0
        solver_options["print_user_options"] = 'yes'
        #solver_options["nlp_scaling_method"] = 'none'
        #solver_options["print_options_documentation"] = 'yes'
        nlp_options = {'ipopt':solver_options}
        S = cas.nlpsol('S', 'ipopt', nlp, nlp_options)
        # Solve the optimisation problem.
        r = S(x0=warm_start, lbg=np.concatenate(constraints_lb), ubg=np.concatenate(constraints_ub))
    else:
        error_str = "Solver " + str(solver) + " is not implemented yet."
        raise NotImplementedError(error_str)
    
    # Extract the solution.
    objective_function = r['f']
    opt_sol = r['x']
    #u = cas.MX.sym('u', agent.input_dim*horizon, 1)
    #x = cas.MX.sym('x', agent.state_dim*(horizon + 1), 1)
    #uc = cas.MX.sym('xc', agent.input_dim, 1)
    #xc = cas.MX.sym('xc', agent.state_dim, 1)
    #yc = cas.MX.sym('yc', agent.output_dim, 1)
    
    # Extract the optimal input sequence.
    u_opt = np.copy(opt_sol[0 : q*horizon])
    # Reshape the optimal input sequence.
    u_opt = np.reshape(u_opt, (q, horizon), order='F')

    # Extract the optimal predicted state sequence.
    x_opt = np.copy(opt_sol[q*horizon : q*horizon + n*(horizon + 1)])
    # Reshape the optimal predicted state sequence.
    x_opt = np.reshape(x_opt, (n, horizon + 1), order='F')
    
    # Extract the optimal cooperation input.
    uc_opt = np.copy(opt_sol[q*horizon + n*(horizon + 1) : q*horizon + n*(horizon + 1) + q])
    
    # Extract the optimal cooperation state.
    xc_opt = np.copy(opt_sol[q*horizon + n*(horizon + 1) + q : q*horizon + n*(horizon + 1) + q + n])
    
    # Extract the optimal cooperation output.
    yc_opt = np.copy(opt_sol[q*horizon + n*(horizon + 1) + q + n : q*horizon + n*(horizon + 1) + q + n + agent.output_dim])
    
    # Return a dictionary with the respective values.
    return {"objective_function": objective_function,
            "u_opt": u_opt,
            "x_opt": x_opt,
            "uc_opt": uc_opt,
            "xc_opt": xc_opt,
            "yc_opt": yc_opt,
            "ipopt_sol": r}   
    

    """Implementation of the MPC problem that each agents solves. The problem is set up in casadi and solved.
    Returns the objective value, optimal input sequence, cooperative output trajectory, and cooperative state and input trajectory as well as the dictionary containing the result of ipopt.
    
    The explicit relation of the cooperative output trajectory and the cooperative state and input trajectory is not used. Instead, it is defined implicitely by adding a suitable constraint to the optimisation problem.
    
    Arguments:
    - agent (mkmpc object): Current agent.
    - horizon (int): Prediction horizon of the MPC optimisation problem.
    - period_length (int): Period length of the cooperative output trajectory.
    - previous_trajectory (array): Trajectory that was previously optimal and from which a deviation is penalised. If none is provided, this part of the cost is excluded, enabling an easy initialisation. (default None)
    - deviation_parameter (double): Parameter that weights the cost on the deviation from the previous cooperation output trajectory. (default 0.001)
    - terminal_cost_matrix (array): Cost matrix that defines the terminal cost and the terminal set as outlined in the paper. If none is provided, a terminal equality constraint is used. (default None)
    - warm_start (list of arrays): Warm start for the MPC problem. If None is provided, zeros are used. The ordering should be:
      sequence of inputs, sequence of states, cooperation input trajectory, cooperation state trajectory, cooperation output trajectory
    - coop_outputs_neighbours (list of list of arrays): Communicated cooperative output trajectories of the neighbours. 
      Ordering corresponds to ordering of 'neighbours' attribute of the agent. If None is provided, the cooperation output trajectories of the neighbours (i.e. of the objects) are taken, which is the intended behaviour.
      If an empty list is provided, the cost for cooperation is not included in the objective.
      This attribute servers mainly for initialisation or tests. (default None)
    - solver (string): One of casadis solver options. Default: ipopt
    """
    # Initialise needed objects.
    constraints = []
    constraints_lb = []
    constraints_ub = []
    objective = cas.MX(0)
    
    # Shorthands.
    n = agent.state_dim
    q = agent.input_dim
    p = agent.output_dim
    T = period_length
    N = horizon
    yT_prev = previous_trajectory
            
    # Create decision variables.
    # Time steps are stacked beneath each other.
    u = cas.MX.sym('u', q*N, 1)  # input sequence
    x = cas.MX.sym('x', n*(N+1), 1)  # state sequence
    uT = cas.MX.sym('uT', q*(T+1), 1)  # input sequence of artificial reference
    xT = cas.MX.sym('xT', n*(T+1), 1)  # state sequence of artificial reference
    yT = cas.MX.sym('yT', p*(T+1), 1)  # output sequence of artificial reference
    
    # Set the initial condition as a constraint.
    constraints.append(x[0 : n, 0] - agent.current_state)
    # This constraint is an equality constraint.
    constraints_lb.append(np.zeros((n,1)))
    constraints_ub.append(np.zeros((n,1)))
    
    # Create constraints containing the dynamics.
    for t in range(N):
        constraints.append(agent.dynamics(x[t*n : (t+1)*n, 0], u[t*q : (t+1)*q, 0]) - x[(t+1)*n : (t+2)*n, 0])
        # These constraints are equality constraints.
        constraints_lb.append(np.zeros((n,1)))
        constraints_ub.append(np.zeros((n,1)))
    # Set the state constraints.
    A = agent.state_constraints["A"]
    b = agent.state_constraints["b"]
    for t in range(N):
        constraints.append(A@x[t*n : (t+1)*n, 0])
        constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
        constraints_ub.append(b)
    # Set the input constraints.
    A = agent.input_constraints["A"]
    b = agent.input_constraints["b"]
    for t in range(N):
        constraints.append(A@u[t*q : (t+1)*q, 0])
        constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
        constraints_ub.append(b)
        
    # Set the terminal constraint.
    if terminal_cost_matrix is None:
        # If no terminal cost matrix is provided, use a terminal equality constraint.
        tau = N%T  # Calculate the step in the T-periodic trajectory at the end of the prediction horizon.
        constraints.append(x[N*n : (N+1)*n, 0] - xT[tau*n : (tau+1)*n, 0])
        constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
        constraints_ub.append(np.zeros((n,1)))
    else:
        raise NotImplementedError('A terminal cost is not implemented yet.')
    
    # Add constraints that link the cooperation state and input sequence to the cooperation output sequence.
    for k in range(T+1):
        constraints.append(agent.output_map(xT[k*n:(k+1)*n, 0], uT[k*q:(k+1)*q, 0]) - yT[k*p:(k+1)*p, 0])
        constraints_lb.append(np.zeros((agent.output_dim,1)))  # This constraint is an equality constraint.
        constraints_ub.append(np.zeros((agent.output_dim,1)))
    # Add a constraint that enforces the cooperation state and input sequence to be a trajectory.
    for k in range(T):
        constraints.append(agent.dynamics(xT[k*n : (k+1)*n, 0], uT[k*q : (k+1)*q, 0]) - xT[(k+1)*n : (k+2)*n, 0])
        constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
        constraints_ub.append(np.zeros((n,1)))
    # constraints.append(agent.dynamics(xT[T*n : (T+1)*n, 0], uT[T*q : (T+1)*q, 0]) - xT[(0)*n : (1)*n, 0])
    # constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
    # constraints_ub.append(np.zeros((n,1)))
    # Add a constraint that enforces the cooperation state and input trajectory to be periodic.
    constraints.append(xT[(T)*n : (T+1)*n , 0] - xT[0*n : 1*n, 0])
    constraints_lb.append(np.zeros((n,1)))  # This constraint is an equality constraint.
    constraints_ub.append(np.zeros((n,1)))
    constraints.append(uT[(T)*q : (T+1)*q , 0] - uT[0*q : 1*q, 0])
    constraints_lb.append(np.zeros((q,1)))  # This constraint is an equality constraint.
    constraints_ub.append(np.zeros((q,1)))
    
    # Set the cooperation output constraints.
    A = agent.cooperation_output_constraint["A"]
    b = agent.cooperation_output_constraint["b"]
    for k in range(T+1):
        constraints.append(A@yT[k*p:(k+1)*p, 0])
        constraints_lb.append(-np.inf*np.ones((b.shape[0], 1)))
        constraints_ub.append(b)
        # Add custom constraints.
        # HINT: We do not want (and need) to consider this yet.
        # constraints.append(agent.custom_output_constraint_function(yT[k*p:(k+1)*p, 0]))
        # constraints_lb.append(-np.Inf*np.ones((1,1)))
        # constraints_ub.append(np.array([[agent.custom_output_constraint_upper_bound]]))
    
    # Create the objective function:
    # Sum up the stage cost.
    stage_cost = agent.stage_cost
    for t in range(horizon):
        tau = t%T  # Calculate the corresponding step in the T-periodic trajectory.
        objective += stage_cost(x[t*n : (t+1)*n, 0], u[t*q : (t+1)*q, 0], xT[tau*n : (tau+1)*n, 0], uT[tau*q : (tau+1)*q, 0])
    # Add the terminal cost.
    #TODO Implement terminal cost.
    # Add the cooperative cost.
    if coop_outputs_neighbours is None:
        for k in range(T):
            for neigh_agent in agent.neighbours:
                yT_neigh = np.copy(neigh_agent.current_cooperation_output)  # Get current cooperation output trajectory of the neighbour.
                # Reshape the neighbours trajectory.
                yT_neigh = np.reshape(yT_neigh, (neigh_agent.output_dim*(period_length+1), 1), order='F')
                # Note: The output dimension of the neighbours should be the same.
                objective += agent.cooperation_cost_summand(yT[k*p:(k+1)*p, 0], yT_neigh[k*p:(k+1)*p, 0])
                objective += neigh_agent.cooperation_cost_summand(yT_neigh[k*p:(k+1)*p, 0], yT[k*p:(k+1)*p, 0])
    elif len(coop_outputs_neighbours):
        raise NotImplementedError('Taking the cooperative output trajectories from a list is not implemented yet.')
    
    # Add the cost that penalises the deviation from a (previous) output trajectory.
    if yT_prev is None:
        # If none was supplied, skip this part of the cost.
        pass
    else:
        for k in range(T):
            objective += deviation_parameter*agent.deviation_cost_summand(yT[k*p:(k+1)*p, 0], yT_prev[(k+1)*p:(k+2)*p, 0])
        objective += deviation_parameter*agent.deviation_cost_summand(yT[(T)*p:(T+1)*p, 0], yT_prev[0*p:1*p, 0])
    
    # Set warm start of solver.
    if warm_start is None:
        # Initialise with zeros if no warm start was provided.
        warm_start = np.zeros((u.shape[0] + x.shape[0] + uT.shape[0] + xT.shape[0] + yT.shape[0], 1))
        # TODO: Initialise the warm start as usual in MPC by the shifted input sequence and reusing the previously optimal solution.
        # Query the agent if time is 0, then use 0.
    
    # Create optimisation object.
    nlp = {'x':cas.vertcat(u, x, uT, xT, yT), 'f':objective, 'g':cas.vertcat(*constraints)}

    solver_options = {}
    if solver is None or solver == 'ipopt':
        #solver_options["fixed_variable_treatment"] = 'make_constraint'
        solver_options["print_level"] = 0
        solver_options["print_user_options"] = 'yes'
        #solver_options["nlp_scaling_method"] = 'none'
        #solver_options["print_options_documentation"] = 'yes'
        nlp_options = {'ipopt':solver_options}
        S = cas.nlpsol('S', 'ipopt', nlp, nlp_options)
        # Solve the optimisation problem.
        r = S(x0=warm_start, lbg=np.concatenate(constraints_lb), ubg=np.concatenate(constraints_ub))
    else:
        error_str = "Solver " + str(solver) + " is not implemented yet."
        raise NotImplementedError(error_str)
    
    # Extract the solution.
    objective_function = r['f']
    opt_sol = r['x']
    #u = cas.MX.sym('u', agent.input_dim*horizon, 1)
    #x = cas.MX.sym('x', agent.state_dim*(horizon + 1), 1)
    #uc = cas.MX.sym('xc', agent.input_dim, 1)
    #xc = cas.MX.sym('xc', agent.state_dim, 1)
    #yc = cas.MX.sym('yc', agent.output_dim, 1)
    
    # Extract the optimal input sequence.
    u_opt = np.copy(opt_sol[0 : q*horizon])
    # Reshape the optimal input sequence.
    u_opt = np.reshape(u_opt, (q, horizon), order='F')
    last_index = q*horizon

    # Extract the optimal predicted state sequence.
    x_opt = np.copy(opt_sol[last_index : last_index + n*(horizon + 1)])
    # Reshape the optimal predicted state sequence.
    x_opt = np.reshape(x_opt, (n, horizon + 1), order='F')
    last_index = last_index + n*(horizon + 1)
    
    # Extract the optimal cooperation input trajectory.
    uT_opt = np.copy(opt_sol[last_index : last_index + q*(T+1)])
    # Reshape the optimal cooperation input trajectory.
    uT_opt = np.reshape(uT_opt, (q, T+1), order='F')
    last_index = last_index + q*(T+1)
    
    # Extract the optimal cooperation state trajectory.
    xT_opt = np.copy(opt_sol[last_index : last_index + n*(T+1)])
    # Reshape the optimal cooperation state trajectory.
    xT_opt = np.reshape(xT_opt, (n, (T+1)), order='F')
    last_index = last_index + n*(T+1)
    
    # Extract the optimal cooperation output trajectory.
    yT_opt = np.copy(opt_sol[last_index : last_index + p*(T+1)])
    # Compute the cost for cooperation.
    cost_for_cooperation = 0.0
    if coop_outputs_neighbours is None:
        for k in range(T):
                for neigh_agent in agent.neighbours:
                    yT_neigh = np.copy(neigh_agent.current_cooperation_output)  # Get current cooperation output trajectory of the neighbour.
                    # Reshape the neighbours trajectory.
                    yT_neigh = np.reshape(yT_neigh, (neigh_agent.output_dim*(period_length+1), 1), order='F')
                    # Note: The output dimension of the neighbours should be the same.
                    cost_for_cooperation += agent.cooperation_cost_summand(yT_opt[k*p:(k+1)*p, 0], yT_neigh[k*p:(k+1)*p, 0])
                    cost_for_cooperation += neigh_agent.cooperation_cost_summand(yT_neigh[k*p:(k+1)*p, 0], yT_opt[k*p:(k+1)*p, 0])
    # Reshape the optimal cooperation output trajectory.
    yT_opt = np.reshape(yT_opt, (p, (T+1)), order='F')
    

    
    # Return a dictionary with the respective values.
    return {"objective_function": objective_function,
            "u_opt": u_opt,
            "x_opt": x_opt,
            "uT_opt": uT_opt,
            "xT_opt": xT_opt,
            "yT_opt": yT_opt,
            "cost_for_cooperation": cost_for_cooperation,
            "ipopt_sol": r}     