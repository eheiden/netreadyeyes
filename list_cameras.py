from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()

print("\nAvailable cameras:\n")

for i, name in enumerate(devices):
    print(f"{i}: {name}")