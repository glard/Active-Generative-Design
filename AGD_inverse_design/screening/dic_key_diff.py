from pymatgen import Composition

# dict1 = {"a": 1, "b": 2, "c": 3, "d": 4, "8": 9}
# dict2 = {"w": "two", "b": "two",  "r": "two", "d": "four", "gk": 89}
# keys1 = dict1. keys()
# keys2 = dict2. keys()
# difference1 = keys1 - keys2
# print(difference1)
#
# difference2 = keys2 - keys1
# print(difference2)
#
# a=(1,2,3,4)
# b=(2,2,5,4)
# print(set(a))
# print(set(b))
# print(set(a) ^ set(b))

def find_diff(a, b):
    # input a, b are tuples
    l = []
    for ai,bi in zip(a,b):
        dic  = {}
        if ai != bi:
            dic[ai] = bi
            l.append(dic)
    return l

#print(find_diff(a,b))


x = 'NaHoCl4'
y = 'BF4Na'
print(Composition(x).as_dict())



x_d = {k: v for k, v in sorted(Composition(x).as_dict().items(), key=lambda item: item[1])}
y_d = {k: v for k, v in sorted(Composition(y).as_dict().items(), key=lambda item: item[1])}

print(x_d)
print(y_d)
x_t = tuple(x_d)
y_t = tuple(y_d)
print(str(find_diff(x_d, y_d)))
