#!/usr/bin/env python

import sys, json

par = json.loads(open("params.json").read())
var = sys.argv[1]
print(par[var])
