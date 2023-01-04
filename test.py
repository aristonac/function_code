# # Define our 3 functions
# def my_function():
#     print("Hello From My Function!")

# def my_function_with_args(username, greeting, age):
#     print("Hello, my name is %s, From My Function!, I wish you %s, my age %s"%(username, greeting, age))

# def sum_two_numbers(a, b):
#     return a + b

# # print(a simple greeting)
# my_function()

# #prints - "Hello, John Doe, From My Function!, I wish you a great year!"
# my_function_with_args("big", "at year!", "18")

# # after this line x will hold the value 3!
# x = sum_two_numbers(1,2)
# print(x)
    
# enumarate list for the selecting index

# a_list = [1,'foo',2,3,4,5,6,7]
# # print(len(a_list))

# for index, elem in enumerate(a_list):

#     if (index+1  < len(a_list) and index >= 0):


#         prev_el = str(a_list[index-1])

#         curr_el = str(elem)

#         next_el = str(a_list[index+1])


#         print(index, 'this is elem', elem)   
        
# enumerate more verstehen

# my_list = ['apple', 'banana', 'grapes', 'pear']
# for c, value in enumerate(my_list,1):
#     print(value,c)

# from itertools import cycle

# li = [0, 1, 2, 3]

# running = True
# licycle = cycle(li)
# # Prime the pump
# nextelem = next(licycle)
# while running:
#     thiselem, nextelem = nextelem, next(licycle)
#     print(nextelem)

# from itertools import tee, islice, chain, zip 

# def previous_and_next(some_iterable):
#     prevs, items, nexts = tee(some_iterable, 3)
#     prevs = chain([None], prevs)
#     nexts = chain(islice(nexts, 1, None), [None])
#     return zip(prevs, items, nexts)

# for previous, item, nxt in previous_and_next(mylist):
#     print "Item is now", item, "next is", nxt, "previous is", previous

li = [0, 1, 2, 3]

running = True
while running:
    for idx, elem in enumerate(li):
        thiselem = elem
        nextelem = li[(idx + 1) % len(li)]
        print (nextelem)