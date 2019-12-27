# -*- coding: utf-8 -*-



import inspect
import tensorflow.compat as package


for name, obj in inspect.getmembers(package):
    if inspect.isclass(obj):
        print(obj)