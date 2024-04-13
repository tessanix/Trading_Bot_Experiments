# http://developers.xstore.pro/documentation/#introduction

# import configparser 
# config = configparser.ConfigParser()
# config.read_file(open('config.ini'))
# # config.read(['config.ini'])
# print(config.sections())


myvars = {}
with open("config.ini") as myfile:
    for line in myfile:
        name, _ , var = line.partition("=")
        # print(f'name:{name}, var:{var}')
        myvars[name] = var.strip('\n')
        # myvars[name.strip()] = float(var)

print(myvars)