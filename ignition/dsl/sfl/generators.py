"""General generators for SFL language"""

import os
import sys

from .language import StrongForm, Variable
from .proteus_coefficient_printer import ProteusCoefficientPrinter
from .proteus_script_printer import ProteusScriptPrinter
from ...code_tools import code_obj
from ...utils.ordered_set import OrderedSet
from ...utils.iterators import flatten


class SFLGenerator(object):
    """Base class for strong form language generator.

    """
    def __init__(self, strong_forms, variables=None, **kwargs):
        self.strong_forms = strong_forms if not isinstance(strong_forms, StrongForm) else (strong_forms,)
        self.variables = variables if not isinstance(variables, Variable) else (variables,)
        self.kwargs = kwargs

    def generate(self):
        pass


class ProteusCoefficientGenerator(SFLGenerator):
    """SFL generator for proteus coefficient evaluation

    """

    def __init__(self, strong_forms, variables=None, **kwargs):
        super(ProteusCoefficientGenerator, self).__init__(strong_forms, variables, **kwargs)
        self._filename = None
        self._classname = None
        self.class_dag = None
        self.modules = OrderedSet()
        self._set_base_class(kwargs.get("base_class"))
        self.gen_evaluate = kwargs.get("gen_evaluate", True)

    @property
    def filename(self):
        if self._filename is None:
            # Try to get filename from keywords
            if "filename" in self.kwargs:
                self._filename = self.kwargs["filename"]
            else:
                # Get filename from script name
                self._filename = os.path.split(sys.argv[0])[1][:-3]+"_proteus.py"
        return self._filename

    @property
    def classname(self):
        if self._classname is None:
            # Try to get class name from keywords
            if "classname" in self.kwargs:
                self._classname = self.kwargs["classname"]
            else:
                # Set class name from script name
                self._classname = os.path.split(sys.argv[0])[1][:-3]\
                    .title().replace("_", "") \
                    + "Coefficients"
        return self._classname

    def _set_base_class(self, base_class):
        if base_class is None:
            self.base_class_name = "TC_base"
            self.base_class_import = "from proteus.TransportCoefficient import TC_base"
        else:
            self.base_class_name = base_class.__name__
            self.base_class_import = "from %s import %s" % (base_class.__module__, base_class.__name__)

    @property
    def module_path(self):
        return os.path.dirname(self.filename)

    @property
    def module_name(self):
        return os.path.basename(self.filename).replace('.py', '')

    def gen_transport_coefficient_dictionary(self):
        if self.variables is None:
            self.variables = OrderedSet(flatten(map(lambda x: x.variables(),
                                                    self.strong_forms)))

        ret_dict = dict([(name, {})
                         for name in StrongForm.transport_eqn_names])
        for sf_idx, sf in enumerate(self.strong_forms):
            transport_coefficients = sf.extract_transport_coefficients()
            for var_idx, variable in enumerate(self.variables):
                for eqn_part, eqn in transport_coefficients.iteritems():
                    eqn_dict = ret_dict[eqn_part].get(sf_idx, {})
                    if eqn_part in ['potential', 'diffusion']:
                        eqn_dict[var_idx] = {var_idx: StrongForm.extract_order(eqn, variable)}
                    else:
                        eqn_dict[var_idx] = StrongForm.extract_order(eqn, variable)
                    ret_dict[eqn_part][sf_idx] = eqn_dict
        return ret_dict

    def gen_init_func_node(self):
        nc = code_obj.Variable("nc", int, var_init=1)
        _M, _A, _B, _C = map(lambda x: code_obj.Variable(x, int, var_init=[0]),
                             ["M", "A", "B", "C"])
        _rFunc = code_obj.Variable("rFunc", "function", var_init=None)
        useSparseDiffusion = code_obj.Variable("useSparseDiffusion",
                                               bool, var_init=True)
        default_input_vars = [_M, _A, _B, _C, _rFunc, useSparseDiffusion]
        inputs = [nc] + default_input_vars
        constructor = self.class_dag.create_constructor(inputs=inputs, variadic=True)

        member_names = ["M", "A", "B", "C"]
        M, A, B, C = map(lambda (x, v): code_obj.IndexedVariable(x, int,
                                                                 var_init=v),
                         zip(member_names, default_input_vars))
        rFunc = code_obj.Variable('rFunc', "function", _rFunc)

        member_vars = [M, A, B, C, rFunc]
        map(lambda x: self.class_dag.add_member_variable(x), member_vars)
        tmp_names = ["mass", "advection", "diffusion", "potential", "reaction",
                     "hamiltonian"]
        mass, advection, diffusion, potential, reaction, hamiltonian = \
            map(lambda name: code_obj.IndexedVariable(name),
                tmp_names)
        tmp_vars = [mass, advection, diffusion, potential, reaction,
                    hamiltonian]
        map(lambda x: constructor.add_object(x), member_vars + tmp_vars)

        coeff_dict = self.gen_transport_coefficient_dictionary()
        for variable, name in zip(tmp_vars, tmp_names):
            constructor.add_statement("=", variable, str(coeff_dict[name]))

        init_args = ["self", nc] + tmp_names + ["useSparseDiffusion = useSparseDiffusion"]
        # Only init TC_Base since I don't have arguments to other base class.
        constructor.add_statement("TC_base.__init__", *init_args)

    def gen_evaluate_func_node(self):
        #XXX: much hardcoded here
        dag = self.class_dag

        t_var = code_obj.Variable('t', int)
        c_var = code_obj.IndexedVariable('c', int)

        nc = code_obj.Variable('nc', int)
        dag.add_member_variable(nc)

        eval_func = code_obj.FunctionNode('evaluate', inputs=(t_var, c_var))
        dag.add_member_function(eval_func)

        eval_loop = code_obj.LoopNode('for', test=nc)
        eval_func.add_object(eval_loop)
        loop_idx = eval_loop.idx
        c_eval_args = (dag.get_member_variable('M').index_stmt(loop_idx),
                       dag.get_member_variable('A').index_stmt(loop_idx),
                       dag.get_member_variable('B').index_stmt(loop_idx),
                       dag.get_member_variable('C').index_stmt(loop_idx),
                       t_var,
                       c_var.index_stmt('x'),
                       c_var.index_stmt("('u', %s)" % loop_idx),
                       c_var.index_stmt("('m', %s)" % loop_idx),
                       c_var.index_stmt("('dm', %s, %s)" % (loop_idx,loop_idx)),
                       c_var.index_stmt("('f', %s)" % loop_idx),
                       c_var.index_stmt("('df', %s, %s)" % (loop_idx,loop_idx)),
                       c_var.index_stmt("('a', %s, %s)" % (loop_idx,loop_idx)),
                       c_var.index_stmt("('r', %s)" % loop_idx),
                       c_var.index_stmt("('dr', %s, %s)" % (loop_idx,loop_idx)),
                       )
        eval_loop.add_statement("self.linearADR_ConstantCoefficientsEvaluate",
                                *c_eval_args)

        # XXX: Total cheat
        if self.kwargs.get("rfunc_flatten"):
            blurb = code_obj.Blurb("""
nSpace=c['x'].shape[-1]
if self.rFunc != None:
    for n in range(len(c[('u',i)].flat)):
        c[('r',i)].flat[n] = self.rFunc[i].rOfUX(c[('u',i)].flat[n],c['x'].flat[n*nSpace:(n+1)*nSpace])
        c[('dr',i,i)].flat[n] = self.rFunc[i].drOfUX(c[('u',i)].flat[n],c['x'].flat[n*nSpace:(n+1)*nSpace])
""")
            eval_func.add_object(blurb)

    def gen_coefficient_class(self, classname=None):
        if classname is not None:
            self._classname = classname
        self.modules.add("import proteus")
        self.modules.add("from proteus.TransportCoefficients import TC_base")
        self.modules.add(self.base_class_import)
        self.class_dag = code_obj.ClassNode(self.classname,
                                            parents=[self.base_class_name])
        if self.gen_evaluate:
            self.class_dag.add_object(code_obj.Blurb("from proteus.ctransportCoefficients import linearADR_ConstantCoefficientsEvaluate"))
        self.gen_init_func_node()
        if self.gen_evaluate:
            self.gen_evaluate_func_node()

    def to_file(self, filename=None):
        self.gen_coefficient_class()
        printer = ProteusCoefficientPrinter(self)
        if filename is not None:
            self._filename = filename
        print("Writing proteus coefficient file to %s" % self.filename)
        printer.print_file(self.filename)


class ProteusScriptGenerator(SFLGenerator):
    """Generates a base script for calling a SFL code from Proteus"""
    def __init__(self, strong_forms, variables=None, **kwargs):
        super(ProteusScriptGenerator, self).__init__(strong_forms, variables,
                                                     **kwargs)
        self.coefficient_class = kwargs.get("coefficient_class")
        self.module_name = kwargs.get("module_name")
        if self.coefficient_class is None:
            self._gen_coefficient_class()
        self._filename = None

    @property
    def filename(self):
        if self._filename is None:
            # Try to get filename from keywords
            if "filename" in self.kwargs:
                self._filename = self.kwargs["filename"]
            else:
                # Get filename from script name
                self._filename = os.path.split(sys.argv[0])[1][:-3]+"_proteus_script.py"
        return self._filename

    def _gen_coefficient_class(self):
        coefficient_generator = ProteusCoefficientGenerator(self.strong_forms,
                                                            self.variables)
        coefficient_generator.to_file()
        self.module_name = coefficient_generator.filename.replace(".py", "")
        self.coefficient_class = coefficient_generator.classname

    def to_file(self, filename=None):
        printer = ProteusScriptPrinter(self)
        if filename is not None:
            self._filename = filename
        print("Writing proteus script file to %s" % self.filename)
        printer.print_file(self.filename)


class UFLGenerator(SFLGenerator):
    """SFL generator for UFL language"""
    def to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.generate())


def generate(framework, strong_forms, variables=None, **kwargs):
    """Generates the equation in a lower level framework"""
    if framework == "proteus-coefficient":
        return ProteusCoefficientGenerator(strong_forms, variables, **kwargs).to_file()
    elif framework == "proteus-script":
        return ProteusScriptGenerator(strong_forms, variables, **kwargs).to_file()
