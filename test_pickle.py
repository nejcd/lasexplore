import pickle

exmple_dict = {1: "3", 2:"4"}

pickle_out = open("dict.pickle", "wb")
pickle.dump(exmple_dict, pickle_out)
pickle_out.close()