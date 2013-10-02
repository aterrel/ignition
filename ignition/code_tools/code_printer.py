"""Code printers from abstract code graph"""

import distutils.ccompiler
import os

import numpy as np

from .code_tools import comment_code, indent_code, NIL
from ..utils.ordered_set import OrderedSet


class CodePrinter(object):
    """Base class for language based code printers"""
    def __init__(self, code_obj, line_comment='', line_end="\n", num_indent=4):
        self.code_obj = code_obj
        self.line_comment = line_comment
        self.line_end = line_end
        self.num_indent = num_indent

    def code_str(self):
        return self._visit_node(self.code_obj)

    def to_file(self, filename, header=None):
        if header:
            self._print_header_file(header)
        with open(filename, 'w') as fp:
            self._print_head(fp)
            fp.write(self.code_str())

    def _print_head(self, fp):
        out_code = "Code generated by Ignition.\n"
        out_code = comment_code(out_code, self.line_comment)
        fp.write(out_code)

    def _print_header_file(self, headername):
        raise RuntimeError("Instance does not implement _print_header_file."
                           "Please instantiate a non-base class.")

    def _visit_codeobj(self, node, indent=0):
        return self._visit_node(node.objs, indent)

    def _visit_blurb(self, node, indent=0):
        return indent_code(node.blurb, indent)

    def _visit_functionnode(self, node, indent=0):
        ret_str = self._decl_func(node)
        ret_str += self._visit_block_head(node, self.num_indent)
        ret_str += self._visit_node(node.expressions, self.num_indent)
        if node.output:
            ret_str +=  self._visit_func_return(node, self.num_indent)
        ret_str += self._visit_block_foot(node, 0)
        return indent_code(ret_str, indent)

    def _visit_indexed_variable(self, node, indent=0):
        return self._visit_variable(node, indent)

    def _visit_loopnode(self, node, indent=0):
        kind = node.kind
        if kind == 'for':
            ret_str = self._visit_for_loop_head(node, indent)
        # TODO: Need to support more while loops
        elif kind == "while":
            ret_str = self._visit_while_loop_head(node, indent)
        else:
            raise NotImplementedError("Do not know how to print %s kind of loop" \
                                      % kind)
        ret_str = ret_str % node.__dict__
        ret_str += self._visit_block_head(node, indent)
        ret_str += self._visit_node(node.expressions, indent)
        if kind == "while":
            ret_str += self._visit_while_loop_inc(node, indent)
        ret_str += self._visit_block_foot(node, indent)
        return indent_code(ret_str, indent)

    def _visit_node(self, node, indent=0):
        if hasattr(node, "__iter__"):
            return "".join(map(lambda n: self._visit_node(n, indent), node))
        if not hasattr(node, "name"):
            return str(node)
        try:
            visitor_func = self.__getattribute__("_visit_%s" % node.name)
        except AttributeError:
            raise RuntimeError("%s does not know how to print %s nodes."
                               % (self.__class__.__name__, node.name))
        return visitor_func(node, indent)

    def _visit_statement(self, node, indent=0, add_end=True):
        operator = node.operator
        args = map(lambda x: self._visit_statement(x, add_end=False)
                   if hasattr(x, 'name') and x.name == "statement" else self._visit_node(x),
                   node.args)
        visit_func_str = "_visit_statement_%s" % node.operator
        if hasattr(self, visit_func_str):
            ret_func = self.__getattribute__(visit_func_str)
        else:
            ret_func = self._visit_statement_default
        ret_str = ret_func(operator, args)
        if add_end:
            ret_str += self.line_end
        return indent_code(ret_str, indent)

    def _visit_statement_default(self, op, args):
        if len(args) == 2 and op in ['=', '+', '+=', '*', '*=', '-', '-=', '/', '/=']:
            ret_str = " ".join([args[0], str(op), args[1]])
        elif len(args) == 1 and op in ['*', '&', '-', '++', '--']:
            ret_str = str(op) + self._visit_node(args[0])
        else:
            ret_str = "%s(%s)" % (op, ", ".join(args))
        return ret_str

    def _visit_statement_index(self, op, args):
        return "%s[%s]" % (args[0], "][".join(map(str, args[1:])))

    def _visit_variable(self, node, indent=0):
        return str(node)


class CCodePrinter(CodePrinter):
    """Code printer for C file"""
    def __init__(self, code_obj):
        super(CCodePrinter, self).__init__(code_obj, line_comment="//",
                                           line_end=";\n", num_indent=2)
        self.c_files = []

    def to_file(self, filename, header=None):
        super(CCodePrinter, self).to_file(filename, header)
        self.c_files.append(filename)

    def to_ctypes_module(self, modname):
        self.to_file(modname+".c", modname+".h")
        self._compile_shared_lib(modname)
        ctypes_mod = "from ctypes import cdll\n"
        ctypes_mod += "lib = cdll.LoadLibrary('./lib%(modname)s.so')\n"
        for f in self.code_obj.functions:
            ctypes_mod += "%(func)s = lib.%(func)s" % {"func": f.func_name}
        with open(modname + ".py", 'w') as fp:
            fp.write(ctypes_mod % {"modname": modname})
        # TODO: Need to manage temporary files better.
        return os.getcwd()

    def _compile_shared_lib(self, libname):
        if not self.c_files:
            raise RuntimeError("Must call to_file before compile")

        # TODO: Add configuration options for compiler
        compiler = distutils.ccompiler.new_compiler()
        objs = compiler.compile(self.c_files)
        compiler.link_shared_lib(objs, libname)

    def _print_header_file(self, headername):
        with open(headername, 'w') as fp:
            self._print_head(fp)

    def _decl_func(self, func_node, add_end=False):
        ret_type = func_node.ret_type
        if ret_type is None:
            ret_type == "void"
        func_args = " ".join(map(lambda x: x.var_type + " " + x.var_name,
                                 func_node.inputs))
        return "%(ret_type)s %(func_name)s(%(func_args)s)%(end)s" \
               % {'ret_type': ret_type,
                  'func_name': func_node.func_name,
                  'func_args': func_args,
                  'end': ';\n' if add_end else '\n',
                  }

    def _decl_index_var(self, var):
        ret_str = "%(var_type)s%(type_mod)s %(var_name)s%(init_str)s"
        var_name = var.var_name
        init_val = var.var_init
        init_str = ''
        type_mod = ''
        if var.shape:
            var_name += "[%s]" % "][".join(map(str, var.shape))
            if init_val is not NIL:
                init_str = " = "
                init_str += str(init_val).replace('[', '{').replace(']', '}')
        else:
            type_mod = "*"
            if init_val:
                raise RuntimeError("Cannot assign init_val to unshaped variable")
        return ret_str % {"var_type": var.var_type,
                          "var_name": var_name,
                          "type_mod": type_mod,
                          "init_str": init_str}

    def _decl_vars(self, vars):
        ret_str = ''
        for var in vars:
            if var.name == "indexed_variable":
                ret_str += self._decl_index_var(var)
            else:
                ret_str += "%(var_type)s %(var_name)s" % var.__dict__
                val = var.var_init
                ret_str += "" if val is NIL else "= %s" % val
            ret_str += ";\n"
        return ret_str

    def _visit_block_head(self, node, indent=0):
        vars = set(node.variables)
        ret_str = "{\n"
        ret_str += indent_code(self._decl_vars(vars), indent)
        return ret_str

    def _visit_block_foot(self, node, indent=0):
        ret_str = "}\n"
        return ret_str

    def _visit_for_loop_head(self, node, indent=0):
        return "for (%(idx)s = %(init)s; " \
               "%(idx)s < %(test)s; %(idx)s += %(inc)s)\n" % node.__dict__

    def _visit_func_return(self, node, indent=0):
        return indent_code("return %(output)s;\n" % node.__dict__, indent)

    def _visit_while_loop_head(self, node, indent=0):
        return "(%(idx)s = %(init)s; \n" \
               "while (%(idx)s < %(test)s)\n" % node.__dict__

    def _visit_while_loop_inc(self, node, indent=0):
        ret_str += indent_code("%(idx)s += %(inc)s;\n" % node.__dict__,
                               self.num_indent)


class PythonCodePrinter(CodePrinter):
    def __init__(self, code_obj):
        super(PythonCodePrinter, self).__init__(code_obj, line_comment="#")
        self.py_files = []
        self.imports = OrderedSet()

    def to_file(self, filename, header=None):
        super(PythonCodePrinter, self).to_file(filename, header)
        self.py_files.append(filename)

    def to_module(self, modname):
        self.to_file(modname + ".py")
        return os.getcwd()

    def _decl_func(self, func_node, add_end=False):
        func_name = func_node.func_name
        if func_name == "<constructor>":
            func_name = "__init__"
        func_args = ""
        inputs = sorted(func_node.inputs, key=lambda x: x.var_init is not None)
        for var in func_node.inputs:
            if var.var_init is not NIL:
                func_args += var.var_name + "=" + repr(var.var_init)
            else:
                func_args += var.var_name
            func_args += ", "
        if func_node.class_member:
            func_args = ", ".join(("self", func_args))
        return "def %(func_name)s(%(func_args)s):\n" \
               % {'func_name': func_name,
                  'func_args': func_args,
                  }

    def _decl_index_var(self, var):
        ret_tmp = "%(var_name)s = %(init_str)s"
        if var.var_init is not NIL:
            if isinstance(var.var_init, np.ndarray):
                self.imports.add("import numpy as np")
                init_str = "np." + repr(var.var_init)
            else:
                init_str = str(var.var_init)
            ret_str = ret_tmp % {"var_name": var.var_name,
                                 "init_str": init_str}
        elif var.shape:
            self.imports.add("import numpy as np")
            init_str = "np.empty(%s)" % "][".join(map(str, var.shape))
            ret_str = ret_tmp % {"var_name": var.var_name,
                                 "init_str": init_str}
        else:
            ret_str = ''
        return ret_str

    def _decl_vars(self, vars):
        ret_str = ''
        for var in vars:
            if var.var_init is NIL:
                continue
            if var.name == "indexed_variable":
                ret_str += self._decl_index_var(var)
            else:
                ret_str += "%(var_name)s = %(val)s" \
                           % {"var_name": self._visit_variable(var),
                              "val": var.var_init}
            ret_str += "\n"
        return ret_str

    def _visit_block_head(self, node, indent=0, decl_vars=None):
        ret_str = ""
        if decl_vars is None:
            vars = set(node.variables)
            ret_str += indent_code(self._decl_vars(vars), self.num_indent)
        else:
            ret_str += indent_code(self._decl_vars(decl_vars), self.num_indent)
        return ret_str

    def _visit_block_foot(self, node, indent=0):
        ret_str = "\n"
        return ret_str

    def _visit_classnode(self, node, indent=0):
        ret_str = "class %(class_name)s(%(parents)s):\n" \
                  % {"class_name": node.class_name,
                     "parents": ",".join(node.parents) if node.parents else 'object',
                     }
        ret_str += self._visit_block_head(node, decl_vars=node.classdict_members)
        ret_str += self._visit_node(node.expressions, self.num_indent)
        ret_str += self._visit_block_foot(node)
        return ret_str

    def _visit_for_loop_head(self, node, indent=0):
        return "for %(idx)s in range(%(init)s, %(test)s, %(inc)s):\n" % node.__dict__

    def _visit_func_return(self, node, indent=0):
        return indent_code("return %(output)s\n" % node.__dict__, indent)


    def _visit_while_loop_head(self, node, indent=0):
        return "(%(idx)s = %(init)s \n" \
               "while (%(idx)s < %(test)s):\n" % node.__dict__

    def _visit_while_loop_inc(self, node, indent=0):
        return indent_code("%(idx)s += %(inc)s\n" % node.__dict__,
                           indent + self.num_indent)

    def _visit_variable(self, node, indent=0):
        ret_str = str(node)
        if node.class_member:
            ret_str = "self." + ret_str
        return ret_str
