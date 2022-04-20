# -*- coding: utf-8 -*-
"""
Functionalities to record matplotlib figures (and other objects).

Sometimes you have already changed your script and you need a lot of work
to get the original figure back. This code records everything that is sent
to a Figure object. When an AxesSubplot object (or any other secondary
object) is created it also automatically records that. When you save the
figure it saves a script at the same time that can completely recreate your
figure. The only limitation is that you do need to make all calls object
oriented. So you shouldn't use any direct function calls to pylab or any
other library. The nice pyik.mplext.cornertext function for instance does
not work but you can add it to the Subplot class as a member function and that
way you can work around this. If something can't be recorded then a warning
is printed and it is documented in the script as comments.

The module records everything that is passed to the matplotlib
Figure objects (by 'wrapping' them) and to save the data as a new python
script. This way the figure can be reproduced if later modifications are
needed.
It is attempted to either textually represent the arguments to the member
functions or to pickle them. If none of this is possible then the recording
process won't succeed.
Althoug in the recorded_figure() function pylab is used to create the initial
figure object, none of the other pylab functionality should be used. I.e. all
function calls should be to the Figure and Ax objects and pylab function
calls won't be recorded.
The ObjectRecorder code may be suitable for other 'recording-business' as
well.

"""
import sys
from pickle import loads, dumps

__all__ = ['recorded_figure', 'recorded_figure_mpl', 'ObjectRecorder']


def recorded_figure(*args, **kwargs):
  """
  Record everything that is done to the wrapped pylab figure and its resulting classes.

  When f.savefig(<fname>) is called a python script with the name <fname>.py
  is saved. This script can later be changed to reproduce the plot in a
  modified version.

  >>> from numpy import *

  >>> f = recorded_figure(figsize=(8,8))
  >>> ax1 = f.add_subplot(121)
  >>> ax2 = f.add_subplot(122)
  >>> ax1.plot([1,2,5,7], color='red');
  l1
  >>> ax2.plot(array([sin(0.1*n) for n in xrange(10)]), color='red');
  l2

  >>> f.savefig('/tmp/test_recorded_figure.pdf')
  """
  from pylab import figure

  return FigureRecorderPylab(figure, known_objects={'figure': figure})(*args, **kwargs)


def recorded_figure_mpl(*args, **kwargs):
  """
  Works the same as recorded_figure() but the wrapped object is a figure created
  with matplotlib only.

  >>> from numpy import *

  >>> f = recorded_figure_mpl(figsize=(8,8))
  >>> ax1 = f.add_subplot(121)
  >>> ax2 = f.add_subplot(122)
  >>> ax1.plot([4,2,5,7], color='red');
  l1
  >>> ax2.plot(array([cos(0.1*n) for n in xrange(10)]), color='red');
  l2

  >>> f.savefig('/tmp/test_recorded_figure_mpl.pdf')
  """
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  from matplotlib.figure import Figure

  def figure(*args, **kwargs):
    f = Figure(*args, **kwargs)
    canvas = FigureCanvas(f)
    return f

  return FigureRecorderMpl(figure, known_objects={'figure': figure})(*args, **kwargs)


class NotRecordable(Exception):
  pass


class ObjectRecorder(object):
  """General Purpose wrapper class that records access to objects."""

  def __init__(self, obj, opcodes=None, names=None, argument_data=None, known_objects=None):
    """Record all function calls and attribute access to obj and allows to save them as a program."""
    if opcodes is None:
      opcodes = list()
    if names is None:
      names = dict()
    if argument_data is None:
      argument_data = list()
    if known_objects is None:
      known_objects = list()
    super(ObjectRecorder, self).__init__()
    self._recorder_object = obj
    self._opcodes = opcodes
    self._names = names
    self._argument_data = argument_data
    if hasattr(obj, '__name__') and obj.__name__ in known_objects and \
       obj == known_objects[obj.__name__]:
      self._recorder_name = obj.__name__
    else:
      basename = self.__simplify_name(type(obj).__name__)
      num = names.get(basename, 0) + 1
      self._names[basename] = num
      self._recorder_name = '%s%s' % (basename, num)
    self._known_objects = known_objects
    self._initialized = True

  def __getattr__(self, attrname):
    retval = getattr(self._recorder_object, attrname)
    if retval:
      retval = type(self)(retval, self._opcodes, self._names,
                          self._argument_data, self._known_objects)
    self._opcodes.append(('__getattr__', self, attrname, retval))
    return retval

  def __clean(self, obj, obj_list=None):
    # to prevent infinite recursion for recursive datatypes
    if not obj_list:
      obj_list = list()
    for other_obj in obj_list:
      if other_obj is obj:
        return obj
    obj_list.append(obj)
    # recursively clean dictionary objects
    if isinstance(obj, dict):
      return dict([(self.__clean(key, obj_list), self.__clean(value, obj_list)) for (key, value) in obj.items()])
    # recursively clean lists
    if isinstance(obj, list) or isinstance(obj, tuple):
      return [self.__clean(x, obj_list) for x in obj]
    # unwrap the object if it is wrapped by an ObjectRecorder object
    if isinstance(obj, ObjectRecorder):
      return obj._recorder_object
    # return the already clean object
    return obj

  def __call__(self, *args, **kwargs):
    retval = self._recorder_object(*self.__clean(args), **self.__clean(kwargs))
    if retval:
      retval = type(self)(retval, self._opcodes, self._names,
                          self._argument_data, self._known_objects)
    self._opcodes.append(('__call__', self, (args, kwargs), retval))
    return retval

  def __setattr__(self, attrname, attrvalue):
    if '_initialized' in self.__dict__:
      self._opcodes.append(('__setattr__', self, (attrname, attrvalue), None))
      setattr(self._recorder_object, attrname, self.__clean(attrvalue))
    else:
      super(ObjectRecorder, self).__setattr__(attrname, attrvalue)

  def __repr__(self):
    self._known_objects[self._recorder_name] = self._recorder_object
    return self._recorder_name

  def __stringify_argument(self, arg, raise_error=True):
    strarg = repr(arg)
    try:
      eval(strarg, dict(), self._known_objects)
    except Exception, e:
      can_be_evaluated = False
    else:
      can_be_evaluated = True
    if len(strarg) < 100:
      is_large = False
    else:
      is_large = True
    try:
      data = dumps(arg)
      loads(data)  # extra precaution to see wether it can also be unpickled
    except Exception, e:
      can_be_pickled = False
    else:
      can_be_pickled = True
    if not can_be_pickled and not can_be_evaluated:
      if raise_error:
        raise NotRecordable(arg)
      else:
        return repr(arg)
    if (can_be_pickled and is_large) or (not can_be_evaluated):
      if data not in self._argument_data:
        self._argument_data.append(data)
      strarg = '__data%s' % self._argument_data.index(data)
    return strarg

  def __string_arguments(self, args, kwargs):
    try:
      return (True, (
        [self.__stringify_argument(arg) for arg in args],
        ["%s=%s" % (name, self.__stringify_argument(arg)) for name, arg in kwargs.items()]
      ))
    except NotRecordable, e:
      sys.stderr.write("WARNING: One of the arguments can't be recorded: %s\n" % e)
      return (False, (
        [self.__stringify_argument(arg, raise_error=False) for arg in args],
        ["%s=%s" % (name, self.__stringify_argument(arg, raise_error=False))
         for name, arg in kwargs.items()]
      ))

  _class_member_replacement_strings = dict()

  def __simplify_name(self, name):
    return self._class_member_replacement_strings.get(name, name)

  def __compress_statements(self, statements):
    lvalues, rvalues, argument_strs, recordable = [list(l) for l in zip(*statements)]
    i = 0
    while True:
      if not i < len(rvalues) - 1:
        break
      if lvalues[i] == rvalues[i + 1] and rvalues[i + 1] not in rvalues[i + 2:] and \
         rvalues[i + 1] not in self._known_objects and recordable[i + 1]:
        lvalues.pop(i)
        rvalues.pop(i + 1)
        recordable.pop(i + 1)
        argument_strs[i] = argument_strs[i] + argument_strs.pop(i + 1)
      i += 1
    for i in xrange(len(lvalues)):
      if lvalues[i] not in rvalues[i + 1:] and \
         lvalues[i] not in self._known_objects:
        lvalues[i] = None
    statements = zip(lvalues, rvalues, argument_strs, recordable)
    return statements

  def __get_statements(self):
    statements = list()
    for op_type, op_obj, op_args, retval in self._opcodes:
      recordable = True
      if op_type == '__call__':
        recordable, (strargs, strkwargs) = self.__string_arguments(*op_args)
        argument_str = '(%s)' % ", ".join(strargs + strkwargs)
      elif op_type == '__getattr__':
        argument_str = '.%s' % op_args
      elif op_type == '__setattr__':
        try:
          value = self.__stringify_argument(op_args[1])
        except NotRecordable:
          value = self.__stringify_argument(op_args[1], raise_error=False)
          recordable = False
        argument_str = '.%s = %s' % (op_args[0], value)
      lvalue = retval._recorder_name if retval else None
      rvalue = op_obj._recorder_name
      statements.append((lvalue, rvalue, argument_str, recordable))
    return self.__compress_statements(statements)

  def _get_recorder_header_code(self):
    lines = [
      "#!/usr/bin/env python",
      "from pickle import dumps, loads"
    ]
    return lines

  def _get_recorder_central_code(self):
    lines = list()
    lines.append('# this method will be called at the end of the script')
    lines.append('def recorded_code():')
    retval_str = 'retval = '
    for lvalue, rvalue, argument_str, recordable in self.__get_statements():
      if lvalue:
        lines.append('  %s%s = %s%s' % (retval_str, lvalue, rvalue, argument_str))
        retval_str = ''
      else:
        lines.append('  %s%s' % (rvalue, argument_str))
      if not recordable:
        lines[-1] = '#' + lines[-1]
    lines.append('  return retval')
    return lines

  def _get_recorder_data_storage_code(self):
    if not self._argument_data:
      return list()
    lines = list()
    lines.append("# The next line%s contain%s too large and non human readable data" %
                 (('', 's') if len(self._argument_data) == 1 else ('s', '')))
    for i, data in enumerate(self._argument_data):
      lines.append("__data%s = loads(%s)" % (i, repr(data)))
    return lines

  def _get_recorder_footer_code(self):
    lines = [
      "returned_object = recorded_code()"
    ]
    return lines

  def get_recorder_code(self):
    """Retrieve the code"""
    lines = (self._get_recorder_header_code() + [""]
             + self._get_recorder_central_code() + [""]
             + self._get_recorder_data_storage_code() + [""]
             + self._get_recorder_footer_code())
    return lines


class FigureRecorder(ObjectRecorder):
  """Recorder object that adds the functionality for the figures to save them as a script."""

  _class_member_replacement_strings = {
    'Figure': 'f',
    'AxesSubplot': 'ax',
    'list': 'l'
  }

  def savefig(self, *args, **kwargs):
    fname = args[0] if args else kwargs['fname']
    super(ObjectRecorder, self).__setattr__('_savefig_fname', fname)
    scriptfile = open('%s.py' % fname, 'w')
    try:
      # if the operating system is unix like make the script executable
      # otherwise... pass the exception silently (since the main
      # functionality won't be affected this seems acceptable here)
      from os import popen
      popen('chmod +x %s.py' % fname)
    except:
      pass
    for line in self.get_recorder_code():
      scriptfile.write('%s\n' % line)
    self._recorder_object.savefig(*args, **kwargs)


class FigureRecorderPylab(FigureRecorder):
  """Recorder object that adds the functionality for the figures to save them as a pylab script."""

  def _get_recorder_header_code(self):
    lines = super(FigureRecorder, self)._get_recorder_header_code()
    lines.append(
      'from pylab import *                                                        '.rstrip(' '))
    lines.append(
      'from optparse import OptionParser                                          '.rstrip(' '))
    return lines

  def _get_recorder_footer_code(self):
    lines = list()
    lines.append(
      'parser = OptionParser("usage: %prog [...]")                          '.rstrip(' '))
    lines.append(
      'parser.add_option("-s", "--show",                                    '.rstrip(' '))
    lines.append('                  action="store_true", dest="show", default=False,   '.rstrip(' '))
    lines.append('                  help="only shows the plot in stead of saving it")  '.rstrip(' '))
    lines.append('(options, args) = parser.parse_args()                                '.rstrip(' '))
    lines.append('if len(args) > 0:                                                    '.rstrip(' '))
    lines.append('  parser.print_help()                                                '.rstrip(' '))
    lines.append('  sys.exit(1)                                                        '.rstrip(' '))
    lines.append('')
    lines += super(FigureRecorder, self)._get_recorder_footer_code()
    lines.append('')
    lines.append('if options.show:                                                     '.rstrip(' '))
    lines.append('  show()                                                             '.rstrip(' '))
    lines.append('else:                                                                '.rstrip(' '))
    lines.append('  returned_object.savefig("%s")                                      '.rstrip(
      ' ') % self._savefig_fname)
    return lines


class FigureRecorderMpl(FigureRecorder):
  """Recorder object that adds the functionality for the figures to save them as a matplotlib script."""

  def _get_recorder_header_code(self):
    lines = super(FigureRecorder, self)._get_recorder_header_code()
    lines.append(
      'from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas'.rstrip(' '))
    lines.append(
      'from matplotlib.figure import Figure                                       '.rstrip(' '))
    lines.append(
      '                                                                           '.rstrip(' '))
    lines.append(
      'def figure(*args, **kwargs):                                               '.rstrip(' '))
    lines.append(
      '  f = Figure(*args, **kwargs)                                              '.rstrip(' '))
    lines.append(
      '  canvas = FigureCanvas(f)                                                 '.rstrip(' '))
    lines.append(
      '  return f                                                                 '.rstrip(' '))
    return lines

  def _get_recorder_footer_code(self):
    lines = super(FigureRecorder, self)._get_recorder_footer_code()
    lines.append('returned_object.savefig("%s")                                              '.rstrip(
      ' ') % self._savefig_fname)
    return lines

if __name__ == "__main__":
  import doctest
  if not doctest.testmod(optionflags=doctest.ELLIPSIS).failed:
    print "Trying to run the created scripts to see wether they're working..."
    from os import popen
    popen('python /tmp/test_recorded_figure.pdf.py')
    popen('python /tmp/test_recorded_figure_mpl.pdf.py')
