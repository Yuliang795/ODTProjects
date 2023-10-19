class connected_component:
  def __init__(self, init_pair, init_excl=None):
    # note init_pair could be only one point, hence * is used here to unpack list elements
    self.cc_point_list = [*init_pair]
    self.excl_cc_set = [init_excl] if init_excl!=None else []

  # input is a single point from one pair of points where the other point is already
  # exists in the cc
  def add_excl_enc(self, encoding):
    self.excl_cc_set.append(encoding)

  def update_cc(self, point):
    self.cc_point_list.append(point)
  def merge_cc(self, obj_cc):
    self.cc_point_list = self.cc_point_list+obj_cc.cc_point_list
    self.excl_cc_set = self.excl_cc_set + obj_cc.excl_cc_set

  def check_excl(self, obj_cc):
    for x in self.excl_cc_set:
      for y in obj_cc.excl_cc_set:
        # if one same encoding in both set, they are exclusive
        if x == y:
          return True
    # if non of the excl encoding matches, return false
    return False


class node_graph:
  def __init__(self, excl_encoder):
    self.cc_set = []
    self.E_plus = []
    self.E_minus = []
    self.excl_encoder = excl_encoder

  # def print_encoder(self):
  #   print(next(self.excl_encoder))

  # update by adding a new point from E+
  # the new pair CANNOT be innor
  def update_cc_set(self,p):
    p0_cc_ind,p1_cc_ind=None,None
    for cc_ind, cc in enumerate(self.cc_set):
      if p[0] in cc.cc_point_list:
        p0_cc_ind = cc_ind
      if p[1] in cc.cc_point_list:
        p1_cc_ind = cc_ind

    # if p0 and p1 both are connected to a component set, merge these two component sets
    if p0_cc_ind !=None and p1_cc_ind !=None:
      self.cc_set[p0_cc_ind].merge_cc(self.cc_set[p1_cc_ind])
      self.cc_set.pop(p1_cc_ind)
      return True

    # if only p0 connected, add the other point to the set where p0 exists
    if p0_cc_ind != None:
      self.cc_set[p0_cc_ind].update_cc(p[1])
      return True
    # if only p1 exists in a set, add the other point to the set where p1 exists
    if p1_cc_ind !=None:
      self.cc_set[p1_cc_ind].update_cc(p[0])
      return True
    # if none of p0 or p1 is in the set, then add a new connected component set for these two points
    self.cc_set.append(connected_component(p))

  # input is a pair of points
  def check_inner(self, p):
    # check only if the connected components set non-empty
    if self.cc_set:
      for cc_ind, cc in enumerate(self.cc_set):
        if p[0] in cc.cc_point_list and p[1] in cc.cc_point_list:
          return True
      # if they are not in the same cc, return False
      return False
    # if the connected components set is empty, return False
    else:
      return False

  # check crossing
  # the input is a pair of points
  def check_crossing(self, p):
    if self.cc_set:
      p0_cc_ind,p1_cc_ind=None,None
      for cc_ind, cc in enumerate(self.cc_set):
        if p[0] in cc.cc_point_list:
          p0_cc_ind = cc_ind
        if p[1] in cc.cc_point_list:
          p1_cc_ind = cc_ind

        # print(f"{cc_ind} -- {cc.cc_point_list}")

      # both of the point have to be in a set, and the sets has to be different
      if p0_cc_ind!=None and p1_cc_ind!=None:
        # print(f"both not non  {p0_cc_ind} - {p1_cc_ind}   ---p {p}")
        # check for different cc set, the two points have to come from two different cc set
        if p0_cc_ind!=p1_cc_ind:
          # check if their cc sets is mutually exclusive
          if self.cc_set[p0_cc_ind].check_excl(self.cc_set[p1_cc_ind]):
              return True
    # if the cc set is empty, or the two point not in two different set, return False
    return False


  # Note: updating the pos edge set will also update the set of connected components
  def update_pos_edge(self, p):
    self.E_plus.append(p)
    # update the cc list after adding new point
    self.update_cc_set(p)

  def update_excl_cc(self, p):

    p0_cc_ind,p1_cc_ind=None,None
    for cc_ind, cc in enumerate(self.cc_set):
      if p[0] in cc.cc_point_list:
        p0_cc_ind = cc_ind
      if p[1] in cc.cc_point_list:
        p1_cc_ind = cc_ind

    # both of the point are in a different set
    # mark the two set as exclusive by adding a unique encoding
    if p0_cc_ind!=None and p1_cc_ind!=None:
      #
        unique_encoding = next(self.excl_encoder)
        self.cc_set[p0_cc_ind].add_excl_enc(unique_encoding)
        self.cc_set[p1_cc_ind].add_excl_enc(unique_encoding)
        return True

    # both of them is new
    # create two new cc sets for each of them
    # and mark the two sets as exclusive
    if p0_cc_ind==None and p1_cc_ind==None:
        unique_encoding = next(self.excl_encoder)
        self.cc_set.append(connected_component([p[0]], unique_encoding))
        self.cc_set.append(connected_component([p[1]], unique_encoding))
        return True

    # only one of the point already exists in one set and the other one isn't
    # add the unique encoding to the set where one of the points exists
    # create a new cc for the other one, and add the unique encoding
    if p0_cc_ind!=None:
        # (1)
        unique_encoding = next(self.excl_encoder)
        self.cc_set[p0_cc_ind].add_excl_enc(unique_encoding)
        # (2)
        self.cc_set.append(connected_component([p[1]], unique_encoding))
        return True

    if p1_cc_ind!=None:
        # (1)
        unique_encoding = next(self.excl_encoder)
        self.cc_set[p1_cc_ind].add_excl_enc(unique_encoding)
        # (2)
        self.cc_set.append(connected_component([p[0]], unique_encoding))
        return True

    # # if the cc set is empty, return False
    # return False

  # update the neg edge and update the exclusive cc set
  def update_neg_edge(self, p):
    self.E_minus.append(p)
    self.update_excl_cc(p)

  # print the cc point list of each cc
  def print_cc(self):
    for cc_ind, cc in enumerate(self.cc_set):
      print(f"connected component {cc_ind}")
      print(cc.cc_point_list)
      print("---exclusive list")
      print(cc.excl_cc_set)
      print("\n")
