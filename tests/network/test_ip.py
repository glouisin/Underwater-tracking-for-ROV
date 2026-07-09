import netifaces 


def get_eth0_ip(iface="eth0"):
    addrs = netifaces.ifaddresses(iface)
    return addrs[netifaces.AF_INET][0]['addr']
print(get_eth0_ip(iface="eth0"))