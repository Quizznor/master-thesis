from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relation, backref
from sqlalchemy import (MetaData, Table, Column, Integer, String, ForeignKey,
                        Unicode, and_, or_)
from sqlalchemy.orm import mapper, relationship, create_session, lazyload
from sqlalchemy.orm.collections import column_mapped_collection, attribute_mapped_collection, mapped_collection


class HirarchicalDatabase:
  """
  Generic hirarchical database class. Used to interface to all database structures.

  Notes
  -----
   Provides functionality, to initialize a hirarchical database,
   based on nodes and attributes.
   New instances of Nodes and Attributes can be accessed via
   provided functions.

  Authors
  -------
   Benjamin Fuchs <benjamin.fuchs@kit.edu>
  """

  engine = create_engine('sqlite:///:memory:', echo=False)
  base = declarative_base()

  class Node(base):
    """
    Basic Node class.

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """

    # name of the table associated with this class
    __tablename__ = 'tnodes'

    # class variable with declaration of corresponding column in table
    # node id, uniqe number to identify that specific node
    nid = Column('nid', Integer, primary_key=True)

    # class variable with declaration of corresponding column in table
    # parent node id, uniqe number to identify the parent node to this one
    # the pid is automatedly filled in from the nodes table
    pid = Column('pid', Integer, ForeignKey('tnodes.nid'))

    # class variable with declaration of corresponding column in table
    # name of this node
    n = Column('n', Unicode)

    # creates a reference to another class/table, in this case the same table this node is associated with
    # necessary to identify parent and child nodes
    # par variable gives reference to parent node. Associated with the node table
    # yields a DB.getNewNode() class instance
    # this is a two way reference, adds to the parent node par, the object chi, which yields a dictionary of
    # DB.getNewNode() class instances, representing the child nodes of the
    # current node. Description of options see below
    par = relationship('Node', collection_class=attribute_mapped_collection('DB.getNewNode.n'), lazy='joined', cascade="all",
                       backref=backref('chi', collection_class=attribute_mapped_collection('n')), remote_side='tnodes.c.nid')

    # creates a reference to another class/table, in this case the attribute table
    # necessary to identify attributes of this node
    # att variable gives reference to attribute , which yields a dictionary of
    # DB.getNewAttribute() class instances, representing the attributes of the current node and is Associated with the node table
    # this is a two way reference, adds to the attribute the object par which yields the parent node, a DB.getNewNode() class instance
    # Description of options see below
    att = relationship('Attribute', collection_class=attribute_mapped_collection(
      'DB.getNewAttribute.n'), lazy='joined', cascade="all, delete-orphan", backref='par')
    # collection_class=attribute_mapped_collection('DB.getNewAttribute.n') : the attributes stored in the att object are returned in a dictionary like object,
    # not as an ordinary list. Can be referenced by the attributes or nodes name.
    # lazy='joined' :  sql internal option, important for tree
    # cascade="all, delete-orphan" : sql internal option, important for tree
    # backref='par' : back reference from the target class to the current
    # class via class object par.

    def __init__(self, name):
      """
      Default node constructor.

      Parameters
      ----------
        name : the name of the node.

      Authors
      -------
       Benjamin Fuchs <benjamin.fuchs@kit.edu>
      """
      self.n = name

    # default string representation of the object, thus the object can be
    # piped to the print function
    def __str__(self):
      """
      String representation of the node for print statements.

      Returns
      -------
        retstr : string representing the node

      Authors
      -------
       Benjamin Fuchs <benjamin.fuchs@kit.edu>
      """
      retstr = "class: <(" + self.__class__.__name__ + ")> has elements: \n"
      for itag in self.__dict__.keys():
        if itag == "n":
          retstr += " '%s = %s' \n" % (itag, self.__dict__[itag])
      return retstr

  # basic attribute class.
  class Attribute(base):
    """
    Basic Attribute class

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """

    # name of the table associated with this class
    __tablename__ = 'tattributes'

    # class variable with declaration of corresponding column in table
    # parent node id, unique number which is the identifier of the associated node
    # the nid is automatedly filled in from the nodes table
    nid = Column('nid', Integer, ForeignKey('tnodes.nid'))

    # class variable with declaration of corresponding column in table
    # attribute id, unique number to identify the attribute in the table
    aid = Column('aid', Integer, primary_key=True)

    # class variable with declaration of corresponding column in table
    # name of the attribute
    n = Column('n', Unicode(100))

    # class variable with declaration of corresponding column in table
    # value of the attribute. If an Integer or Boolean value is passed it is
    # implicitly converted to Unicode-String for the table, and implicitly
    # converted back to the original type when returned
    v = Column('v', Unicode(255))

    # default constructor. An attribute name and value has to be given
    def __init__(self, name, value):
      """
      Default attribute constructor.

      Parameters
      ----------
        name : the name of the attribute.
        value : the value of the attribute.

      Authors
      -------
       Benjamin Fuchs <benjamin.fuchs@kit.edu>
      """
      self.n = name
      self.v = value

    # default string representation of the object, thus the object can be
    # piped to the print function
    def __str__(self):
      """
      String representation of the Attribute for print statements.

      Returns
      -------
        retstr : string representing the Attribute

      Authors
      -------
       Benjamin Fuchs <benjamin.fuchs@kit.edu>
      """
      retstr = "class <(" + self.__class__.__name__ + ")> has elements: \n"
      for itag in self.__dict__.keys():
        if itag in ["n", "v"]:
          retstr += " '%s = %s' \n" % (itag, self.__dict__[itag])
      return retstr

  def __init__(self, eng=None):
    """
    Set up of the SQL engine (connection to the sqlite or mysql database).
    Intializes also the mapper, which keeps the python objects in sync with the database

    Parameters
    ----------
      eng : optional, a string representing the enginge: 'sqlite:///:memory:'

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """

    if eng == None:
      eng = 'sqlite:///:memory:'

    if type(eng) != type(str()):
      raise ValueError("The engine has to be specified by a string! Example: sqlite:///:memory:")

    # create connection to sqlite database in memory
    self.engine = create_engine(eng, echo=False)

    # initialize the mapper, which will map the classes to tables in the database
    self.base.metadata.bind = self.engine

    # creating all tables associated with the classes, the database is initialized
    self.base.metadata.create_all(self.engine)

  def getNewNode(self, name):
    """
    Returns a new node instance.

    Returns
    -------
      a new Node instance

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """
    return self.Node(name)

  def getEngine(self):
    """
    Returns the engine associated with the database.

    Returns
    -------
      the engine

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """
    return self.engine

  def getNewAttribute(self, name, value):
    """
    Returns a new attribute instance

    Returns
    -------
      a new Attribute instance

    Authors
    -------
     Benjamin Fuchs <benjamin.fuchs@kit.edu>
    """
    return self.Attribute(name, value)


def dbGetDown(beg, intend=0):
  """
  prints recursive down the contents of the tree

  Parameters
  ----------
  beg : starting node of the tree
  intend : optional, starting intendation level for displaying

  Returns
  -------
  retstr : a string representation of the tree

  Authors
  -------
  Benjamin Fuchs <benjamin.fuchs@kit.edu>
  """
  if beg.chi == None:
    return ""
  else:
    retstr = " " * intend + "|--+-" + beg.n + "\n"
    intend += 3
    for i in beg.att.keys():
      retstr += " " * intend + "|." + i + " : " + unicode(beg.att[i].v) + "\n"
    for ichi in beg.chi:
      retstr += dbGetDown(beg.chi[ichi], intend)
    return retstr

# recursive upward tree print function returns a string!


def dbGetUp(beg, intend=0):
  """
  prints recursive up the contents of the tree

  Parameters
  ----------
  beg : starting node of the tree
  intend : optional, starting intendation level for displaying

  Returns
  -------
  retstr : a string representation of the tree

  Authors
  -------
  Benjamin Fuchs <benjamin.fuchs@kit.edu>
  """
  if beg.par == None:
    return ""
  else:
    retstr = dbGetUp(beg.par, intend + 3) + " " * intend + "|--+-" + beg.n + "\n"
    intend += 3
    for i in beg.att.keys():
      retstr += " " * intend + "|." + i + " : " + unicode(beg.att[i].v) + "\n"
    return retstr


if __name__ == "__main__":

  # EXAMPLE START!

  # start of example program.
  # BE AWARE THAT THIS SCRIPT NEEDS sqlalchemy version 0.6 beta3 or above!

  # create a new HirarchicalDatabase object
  DB = HirarchicalDatabase()

  # create a session, an interface to the database, through which all transactions are done.
  Session = sessionmaker()

  # configure the session, binding the database to the session
  Session.configure(bind=DB.getEngine())

  # creating a session instance as interface to the database.
  ses = Session()

  # creating a root node with name REAS3
  par = DB.getNewNode("REAS3")

  # adding the root node to the database.
  # all tables are updated. When the class changes, all tables are also updated.
  # thus only one ses.add() call is necessary
  ses.add(par)

  # creating a child node
  chi = DB.getNewNode("3388350")

  # adding the child node to the root node
  par.chi["3388350"] = chi

  # saving the child node as the new parent node
  par = chi

  # creating a new child node
  chi = DB.getNewNode("0")

  # adding an attribute to the child, named hasANG with bool value True
  chi.att["hasANG"] = DB.getNewAttribute("hasANG", True)

  # adding an attribute to the child
  chi.att["hasST"] = DB.getNewAttribute("hasST", True)

  # adding an attribute to the child
  chi.att["hasCH"] = DB.getNewAttribute("hasCH", False)

  # adding an attribute to the child
  chi.att["AZI"] = DB.getNewAttribute("AZI", 5)

  # adding an attribute to the child
  chi.att["ZEN"] = DB.getNewAttribute("ZEN", 10)

  # adding the child and his attributes attribute to the parent
  par.chi["0"] = chi

  # saving the child node as the new parent node
  par = chi

  # creating a new child node
  chi = DB.getNewNode("ST")

  # adding the child to the parent
  par.chi["ST"] = chi

  # saving the child node as the new parent node
  par = chi

  # creating a new child node
  chi = DB.getNewNode("0")

  # adding an attribute to the child
  chi.att["TS"] = DB.getNewAttribute("TS", "./test.root")

  # adding an attribute to the child
  chi.att["SPEC"] = DB.getNewAttribute("TS", "./test.root")

  # adding the child and his attributes attribute to the parent
  par.chi["0"] = chi

  # adding a new root node, details as above
  par = DB.getNewNode("REAL")
  ses.add(par)
  chi = DB.getNewNode("3388350")
  par.chi["3388350"] = chi
  par = chi
  chi = DB.getNewNode("0")
  chi.att["hasANG"] = DB.getNewAttribute("hasANG", True)
  chi.att["hasST"] = DB.getNewAttribute("hasST", False)
  chi.att["hasCH"] = DB.getNewAttribute("hasCH", True)
  chi.att["AZI"] = DB.getNewAttribute("AZI", 15)
  chi.att["ZEN"] = DB.getNewAttribute("ZEN", 90)
  par.chi["0"] = chi
  par = chi
  chi = DB.getNewNode("CH")
  par.chi["CH"] = chi
  par = chi
  chi = DB.getNewNode("1")
  par.chi["1"] = chi
  chi.att["TS"] = DB.getNewAttribute("TS", "./test.root")

  # adding a 3rd, new root node, details as above
  par = DB.getNewNode("MGRM")
  ses.add(par)
  chi = DB.getNewNode("3388350")
  par.chi["3388350"] = chi
  par = chi
  chi = DB.getNewNode("0")
  chi.att["hasANG"] = DB.getNewAttribute("hasANG", True)
  chi.att["hasST"] = DB.getNewAttribute("hasST", False)
  chi.att["hasCH"] = DB.getNewAttribute("hasCH", True)
  chi.att["AZI"] = DB.getNewAttribute("AZI", 90)
  chi.att["ZEN"] = DB.getNewAttribute("ZEN", 20)
  par.chi["0"] = chi
  par = chi
  chi = DB.getNewNode("CH")
  par.chi["CH"] = chi
  par = chi
  chi = DB.getNewNode("0")
  par.chi["0"] = chi
  chi.att["TS"] = DB.getNewAttribute("TS", "./test.root")

  # example database selection/search:
  # query(DB.getNewNode)            : defines which informations are returned by the query
  # filter()             : defines a filter function on the table defined by query(),
  #                       join statements. In this case on the node table
  # join()               : joins to the current table the element "par" of the current node,
  #                       which is basically another node
  # aliased=True         : adds a unique identifier as name to the table associated with par.
  #                       Relevant since the node table is allready
  #                       loaded via the query(DB.getNewNode) statement, and is joined with himself.
  # from_joinpoint=True  : joins the second statement to the last one, not additionally to
  #                       the query() statement as the last join. Thus we crawl up the tree
  # first()              : yields the first table element (as a node class instance) that matches the
  #                       statement. Besides first() there exists a function all() which yeilds an array
  #                       of node instances. If no match is found, None is returned
  res = ses.query(DB.Node).filter(DB.Node.n == "0").\
      join('par', aliased=True).filter(DB.Node.n == '3388350').\
      join('par', aliased=True, from_joinpoint=True).filter(DB.Node.n == 'REAS3').\
      first()

  # printing the tree from the sql query result
  print dbGetUp(res, 0)

  # selecting the branch REAS3 in the tree
  res = ses.query(DB.Node).filter(DB.Node.n == "REAS3").first()

  # printing the tree from the sql query result
  print dbGetDown(res, 0)

  # checking if a node named 3388350 exists in both branches REAS3, and REAL
  # if a match is found, both node instances for 3388350 are returned in an
  # array [DB.getNewNode(REAS3_3388350), DB.getNewNode(REAL_3388350)]
  res = ses.query(DB.Node).filter(DB.Node.n == "3388350").\
      join('par', aliased=True).filter(or_(DB.Node.n == 'REAS3', DB.Node.n == 'REAL')).\
      all()

  # print both trees of the result from the last querry
  print dbGetDown(res[0].par, 0)
  print dbGetUp(res[1].par, 0)
