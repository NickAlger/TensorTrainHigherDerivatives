import numpy as np
from fenics import *
from ufl import replace
from collections_extended import bag, frozenbag

class HigherDerivativesAction:
    def __init__(me, quantity_of_interest_form_Q, state_equation_S, state_bcs,
                 parameter_m, state_variable_u):
        me.Q = quantity_of_interest_form_Q
        me.S = state_equation_S
        me.m = parameter_m
        me.u = state_variable_u

        me.uu = {me.u : frozenbag()}
        me.iuu = {frozenbag(): me.u}

        me.bcs = state_bcs
        me.zero_bcs = [DirichletBC(bc) for bc in me.bcs]
        for bc in me.zero_bcs:
            bc.homogenize()

        me.SS = {me.S : frozenbag()}
        me.iSS = {frozenbag() : me.S}

        me.QQ = {me.Q : frozenbag()}
        me.iQQ = {frozenbag() : me.Q}

        me.M = me.m.function_space()
        me.V = me.u.function_space()
        me.Z = me.Q.arguments()[0].function_space()

        me.v = Function(me.V) # Adjoint variable
        me.z = Function(me.Z)
        S2 = replace(me.S, {TestFunction(me.V) : me.v})
        Q2 = replace(me.Q, {TestFunction(me.Z) : me.z})
        lagrangian = Q2 + S2

        me.A = derivative(lagrangian, me.u, TestFunction(me.V)) # Adjoint equation
        me.G = derivative(lagrangian, me.m, TestFunction(me.M))

        me.vv = {me.v : frozenbag()}
        me.ivv = {frozenbag() : me.v}

        me.AA = {me.A : frozenbag()}
        me.iAA = {frozenbag() : me.A}

        me.GG = {me.G : frozenbag()}
        me.iGG = {frozenbag() : me.G}

        me.solved_variables = set()

        me.all_right_hand_sides = dict() # equation -> rhs form

        me.__incremental_state_solver, me.__incremental_adjoint_solver = me.__make_solvers()

        #

        n_hash_vecs = 10
        me.__M_hash_vecs = np.random.randn(n_hash_vecs, me.M.dim())
        # me.__V_hash_vecs = np.random.randn(n_hash_vecs, me.V.dim())
        me.__Z_hash_vecs = np.random.randn(n_hash_vecs, me.Z.dim())

        me.__m_hash = me.__compute_function_hash(me.m)
        me.__z_hash = me.__compute_function_hash(me.z)
        me.__pp_hashes = dict() # derivative direction p -> hash vector
        me.__hash_tol = 1e-14

        me.num_changed_p = 0

    ####

    def __reset_variable(me, w):
        # w.vector().zero()
        me.solved_variables.discard(w)

    def __reset_all_state_variables(me):
        for u in me.uu.keys():
            me.__reset_variable(u)

    def __reset_all_adjoint_variables(me):
        for v in me.vv.keys():
            me.__reset_variable(v)

    def __reset_p_dependencies(me, p):
        for w in me.__derivative_direction_dependencies(p):
            me.__reset_variable(w)

    def __derivative_direction_dependencies(me, p):
        dependencies_of_p = set()

        for qq in me.iuu.keys():
            for q in qq.unique_elements():
                if q == p:
                    dependencies_of_p.add(me.iuu[qq])
                    break

        for qq in me.ivv.keys():
            for q in qq.unique_elements():
                if q == p:
                    dependencies_of_p.add(me.ivv[qq])
                    break

        return dependencies_of_p

    def __compute_function_hash(me, w):
        w_vec = w.vector()[:].reshape(-1)
        if w == me.m:
            h = np.dot(me.__M_hash_vecs, w_vec)
        elif w == me.z:
            h = np.dot(me.__Z_hash_vecs, w_vec)
        else:
            h = np.dot(me.__M_hash_vecs, w_vec)
        return h

    def __check_if_m_changed_and_update_accordingly(me):
        new_hash = me.__compute_function_hash(me.m)
        old_hash = me.__m_hash
        function_changed = np.linalg.norm(new_hash - old_hash) > me.__hash_tol
        if function_changed:
        # if True:
            me.__m_hash = new_hash
            me.__reset_all_state_variables()
            me.__reset_all_adjoint_variables()
            me.__incremental_state_solver, me.__incremental_adjoint_solver = me.__make_solvers()

    def __check_if_z_changed_and_update_accordingly(me):
        new_hash = me.__compute_function_hash(me.z)
        old_hash = me.__z_hash
        function_changed = np.linalg.norm(new_hash - old_hash) > me.__hash_tol
        if function_changed:
            me.__z_hash = new_hash
            me.__reset_all_adjoint_variables()

    def __check_if_p_changed_and_update_accordingly(me, p):
        new_hash = me.__compute_function_hash(p)
        if p in me.__pp_hashes:
            old_hash = me.__pp_hashes[p]
            function_changed = np.linalg.norm(new_hash - old_hash) > me.__hash_tol
            if function_changed:
            # if True:
                me.__pp_hashes[p] = new_hash
                me.__reset_p_dependencies(p)
                me.num_changed_p = me.num_changed_p + 1
        else:
            me.__pp_hashes[p] = new_hash
            # print('New p')

    ####

    def __make_solvers(me):
        # Perform nonlinear solve of state equation for u
        solve(me.S == 0, me.u, bcs=me.bcs, solver_parameters={"newton_solver": {"maximum_iterations": 50}})
        me.solved_variables.add(me.u)

        # Construct linear solver for incremental state equations
        S_matrix = assemble(lhs(derivative(me.S, me.u, TrialFunction(me.V))))
        for bc in me.bcs:
            bc.apply(S_matrix)

        incremental_state_solver = LUSolver(S_matrix)

        # Linear solver for incremental adjoint equations
        A_matrix = assemble(lhs(derivative(me.A, me.v, TrialFunction(me.V))))
        for bc in me.zero_bcs:
            bc.apply(A_matrix)

        incremental_adjoint_solver = LUSolver(A_matrix)

        return incremental_state_solver, incremental_adjoint_solver

    ####

    def __variable_context(me, w):
        if w in me.uu:
            return me.uu, me.iuu, me.SS, me.iSS
        elif w in me.vv:
            return me.vv, me.ivv, me.AA, me.iAA
        else:
            raise RuntimeError('variable must be in uu or vv')

    def __equation_context(me, E):
        if E in me.SS:
            return me.uu, me.iuu, me.SS, me.iSS
        elif E in me.AA:
            return me.vv, me.ivv, me.AA, me.iAA
        else:
            raise RuntimeError('equation must be in SS or AA')

    def __form_context(me, F):
        if F in me.SS:
            return me.SS, me.iSS
        elif F in me.AA:
            return me.AA, me.iAA
        elif F in me.QQ:
            return me.QQ, me.iQQ
        elif F in me.GG:
            return me.GG, me.iGG
        else:
            raise RuntimeError('form must be in SS or AA or QQ or GG')

    ####

    def __derivative_of_variable(me, w, p):
        ww, iww, _, _ = me.__variable_context(w)
        pp = ww[w] + frozenbag([p])
        if pp in iww:
            dwdp = iww[pp]
        else:
            dwdp = Function(w.function_space())
            ww[dwdp] = pp
            iww[pp] = dwdp
        return dwdp

    def __derivative_of_form(me, F, p):
        FF, iFF = me.__form_context(F)
        pp = FF[F] + frozenbag([p])
        if pp in iFF:
            dFdp = iFF[pp]
        else:
            dFdp = derivative(F, me.m, p)
            for w in F.coefficients():
                if (w in me.uu) or (w in me.vv):
                    dFdp = dFdp + derivative(F, w, me.__derivative_of_variable(w, p))
            FF[dFdp] = pp
            iFF[pp] = dFdp
        return dFdp

    def __form_derivative_recursive(me, pp, FF, iFF):
        if pp in iFF:
            dFdpp = iFF[pp]
        else:
            pp0 = bag(pp)
            p = pp0.pop()
            pp0 = frozenbag(pp0)
            dFdpp0 = me.__form_derivative_recursive(pp0, FF, iFF)
            dFdpp = me.__derivative_of_form(dFdpp0, p)
            FF[dFdpp] = pp
            iFF[pp] = dFdpp
        return dFdpp

    ####

    def _equation_for_variable(me, w):
        ww, _, EE, iEE = me.__variable_context(w)
        pp = ww[w]
        if pp in iEE:
            E = iEE[pp]
        else:
            E = me.__form_derivative_recursive(pp, EE, iEE)
        return E

    def _variable_for_equation(me, E):
        _, iww, EE, _ = me.__equation_context(E)
        w = iww[EE[E]]
        return w

    def _right_hand_side_of_equation(me, E):
        if E in me.all_right_hand_sides:
            E_rhs = me.all_right_hand_sides[E]
        else:
            w = me._variable_for_equation(E)
            E_rhs = rhs(replace(E, {w:TrialFunction(w.function_space())}))
            me.all_right_hand_sides[E] = E_rhs
        return E_rhs

    def _variables_that_form_depends_on(me, F):
        return {w for w in F.coefficients() if (w in me.uu) or (w in me.vv)}

    def _variables_that_equation_rhs_depends_on(me, E):
        w = me._variable_for_equation(E)
        dependencies = me._variables_that_form_depends_on(E)
        dependencies.remove(w)
        return dependencies

    ####

    def _solve_for_variable_recursive(me, w):
        if w not in me.solved_variables:
            E = me._equation_for_variable(w)
            for x in me._variables_that_equation_rhs_depends_on(E):
                me._solve_for_variable_recursive(x)
            me.__solve_incremental_equation(E)
            me.solved_variables.add(w)

    def __solve_incremental_equation(me, E):
        x = me._variable_for_equation(E)
        b = assemble(me._right_hand_side_of_equation(E))
        for bc in me.zero_bcs:
            bc.apply(b)

        if x in me.uu:
            me.__incremental_state_solver.solve(x.vector(), b)
        elif x in me.vv:
            me.__incremental_adjoint_solver.solve(x.vector(), b)
        else:
            raise RuntimeError('can only solve for variable in uu or vv')

    def _evaluate_form(me, F):
        ww = me._variables_that_form_depends_on(F)
        for w in ww:
            me._solve_for_variable_recursive(w)
        return assemble(F)

    ####

    def compute_derivative_of_quantity_of_interest(me, derivative_directions_pp, output_direction_z):
        pp = frozenbag(derivative_directions_pp)
        if output_direction_z is not None:
            me.z.vector()[:] = output_direction_z.vector()[:].copy()

        me.__check_if_m_changed_and_update_accordingly()
        me.__check_if_z_changed_and_update_accordingly()
        me.num_changed_p = 0
        for p in pp.unique_elements():
            me.__check_if_p_changed_and_update_accordingly(p)
        # print('me.num_changed_p=', me.num_changed_p)

        # me.solved_variables.clear()

        if output_direction_z is not None:
            F = me.__form_derivative_recursive(pp, me.GG, me.iGG)
        else:
            F = me.__form_derivative_recursive(pp, me.QQ, me.iQQ)

        return me._evaluate_form(F)
