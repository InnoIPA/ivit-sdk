 import onnx                                                                                        
  m = onnx.load('/home/dos/.cache/ivit/models/efficientnet-b0.onnx')                                 
  nodes = m.graph.node                                                                               
  print(f'total nodes: {len(nodes)}')                                                                
  from collections import Counter                                                                    
  ops = Counter(n.op_type for n in nodes)                                                            
  print('top ops:', dict(ops.most_common(10)))
  print()                                                                                            
  # input/output                                               
  for x in m.graph.input:                                                                            
      dims = [d.dim_value or d.dim_param or '?' for d in x.type.tensor_type.shape.dim]
      print(f'INPUT  {x.name}: {dims}')                                                              
  for x in m.graph.output:                                     
      dims = [d.dim_value or d.dim_param or '?' for d in x.type.tensor_type.shape.dim]               
      print(f'OUTPUT {x.name}: {dims}')                                                              
  print()
  # Find every GAP and its input spatial                                                             
  import onnx.shape_inference as si                            
  m2 = si.infer_shapes(m)                                                                            
  shape_map = {}
  for vi in list(m2.graph.value_info) + list(m2.graph.input) + list(m2.graph.output):                
      dims = []                                                                                      
      for d in vi.type.tensor_type.shape.dim:
          dims.append(d.dim_value if d.dim_value > 0 else None)                                      
      shape_map[vi.name] = dims                                
                                                                                                     
  print('=== GlobalAveragePool / Reduce-style pools ===')
  gap_nodes = [n for n in nodes if n.op_type in ('GlobalAveragePool', 'AveragePool', 'ReduceMean')]  
  for n in gap_nodes:                                                                                
      inp_shape = shape_map.get(n.input[0])
      out_shape = shape_map.get(n.output[0])                                                         
      print(f'  [{n.op_type}] {n.name}  in={inp_shape}  out={out_shape}')  