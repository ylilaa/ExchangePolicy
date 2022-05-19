def neer(baseUS, baseEU, USXR, EUXR):
    temp1 = pow((USXR/baseUS),0.4)
    temp2 = pow((EUXR/baseEU),0.6)
    result = temp1*temp2
    return result 