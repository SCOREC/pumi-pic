for ptcls = [5 10 100 1000]
  tname = strcat('largeE_',int2str(ptcls))
  tname = strcat(tname,'P')
  generate_graphs(tname)
  tname = strcat(tname,'_Optimal')
  generate_graphs(tname)
end
