from logic.prop import Form

lst = [
    'Cp1p2',
    'CCp1p2p3',
    'Cp1Dp2p3',
    'Ip1Dp2Np3',
    'Dp1Ip2Np3',
    #'Ip1Dp2Np3p4',
    'CIp1p2p3'

]
for l in lst:
    f = Form.parse_prefix(l)
    print(l, f)
