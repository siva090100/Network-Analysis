# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:34 2019

@author: HP
"""

import dpkt, pcap
pc = pcap.pcap()     # construct pcap object
pc.setfilter('icmp') # filter out unwanted packets
for timestamp, packet in pc:
    print dpkt.ethernet.Ethernet(packet)