% MATLAB file for graph generation
clear

% Excluded: CSR, Even Distribution

%% Data Reading
fileID_rebuild = fopen('data/smallE_largeP_rebuild.dat');
fileID_push = fopen('data/smallE_largeP_push.dat');
fileID_migrate = fopen('data/smallE_largeP_migrate.dat');

% struct, element_number, distribution, particles_moved, average_time
rebuild_data = fscanf(fileID_rebuild, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_push);
migrate_data = fscanf(fileID_migrate, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_migrate);

%% Data Filtering

% find length of graphs
elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
scs_length = length(unique(rebuild_data( rebuild_data(:,1) == 0, 2 )));
%csr_length = length(unique(rebuild_data( rebuild_data(:,1) == 1, 2 )));
cabm_length = length(unique(rebuild_data( rebuild_data(:,1) == 2, 2 )));

% Only take instances with 50% particles moved, pull distribution and time
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0,[3,5] );
%csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, [3,5] );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, [3,5] );
scs_push = push_data( push_data(:,1) == 0, [3,5] );
%csr_push = push_data( push_data(:,1) == 1, [3,5] );
cabm_push = push_data( push_data(:,1) == 2, [3,5] );
scs_migrate = migrate_data( migrate_data(:,1) == 0, [3,5] );
%csr_migrate = migrate_data( migrate_data(:,1) == 1, [3,5] );
cabm_migrate = migrate_data( migrate_data(:,1) == 2, [3,5] );

% Separate data by distribution, {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% SCS Rebuild
%scs_rebuild_even = scs_rebuild( scs_rebuild(:,1) == 0, 2);
scs_rebuild_uni = scs_rebuild( scs_rebuild(:,1) == 1, 2);
scs_rebuild_gauss = scs_rebuild( scs_rebuild(:,1) == 2, 2);
scs_rebuild_exp = scs_rebuild( scs_rebuild(:,1) == 3, 2);
% % CSR Rebuild
% %csr_rebuild_even = csr_rebuild( csr_rebuild(:,1) == 0, 2);
% csr_rebuild_uni = csr_rebuild( csr_rebuild(:,1) == 1, 2);
% csr_rebuild_gauss = csr_rebuild( csr_rebuild(:,1) == 2, 2);
% csr_rebuild_exp = csr_rebuild( csr_rebuild(:,1) == 3, 2);
% CabM Rebuild
%cabm_rebuild_even = cabm_rebuild( cabm_rebuild(:,1) == 0, 2);
cabm_rebuild_uni = cabm_rebuild( cabm_rebuild(:,1) == 1, 2);
cabm_rebuild_gauss = cabm_rebuild( cabm_rebuild(:,1) == 2, 2);
cabm_rebuild_exp = cabm_rebuild( cabm_rebuild(:,1) == 3, 2);

% SCS Pseudo-Push
%scs_push_even = scs_push( scs_push(:,1) == 0, 2);
scs_push_uni = scs_push( scs_push(:,1) == 1, 2);
scs_push_gauss = scs_push( scs_push(:,1) == 2, 2);
scs_push_exp = scs_push( scs_push(:,1) == 3, 2);
% % CSR Pseudo-Push
% %csr_push_even = csr_push( csr_push(:,1) == 0, 2);
% csr_push_uni = csr_push( csr_push(:,1) == 1, 2);
% csr_push_gauss = csr_push( csr_push(:,1) == 2, 2);
% csr_push_exp = csr_push( csr_push(:,1) == 3, 2);
% CabM Pseudo-Push
%cabm_push_even = cabm_push( cabm_push(:,1) == 0, 2);
cabm_push_uni = cabm_push( cabm_push(:,1) == 1, 2);
cabm_push_gauss = cabm_push( cabm_push(:,1) == 2, 2);
cabm_push_exp = cabm_push( cabm_push(:,1) == 3, 2);

% SCS Migrate
%scs_migrate_even = scs_migrate( scs_migrate(:,1) == 0, 2);
scs_migrate_uni = scs_migrate( scs_migrate(:,1) == 1, 2);
scs_migrate_gauss = scs_migrate( scs_migrate(:,1) == 2, 2);
scs_migrate_exp = scs_migrate( scs_migrate(:,1) == 3, 2);
% % CSR Migrate
% %csr_migrate_even = csr_migrate( csr_migrate(:,1) == 0, 2);
% csr_migrate_uni = csr_migrate( csr_migrate(:,1) == 1, 2);
% csr_migrate_gauss = csr_migrate( csr_migrate(:,1) == 2, 2);
% csr_migrate_exp = csr_migrate( csr_migrate(:,1) == 3, 2);
% CabM Migrate
%cabm_migrate_even = cabm_migrate( cabm_migrate(:,1) == 0, 2);
cabm_migrate_uni = cabm_migrate( cabm_migrate(:,1) == 1, 2);
cabm_migrate_gauss = cabm_migrate( cabm_migrate(:,1) == 2, 2);
cabm_migrate_exp = cabm_migrate( cabm_migrate(:,1) == 3, 2);

%% Graph Generation

% SCS Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'SCS Average Function Times')
xlabel(t, {'Number Particles (Ten Thousands)','Number Elements'})
ylabel(t, 'Seconds')
% Even (Excluded)
% Uniform
ax2 = nexttile;
tile2 = area(elms, [scs_push_uni, scs_rebuild_uni, scs_migrate_uni]);
ax = gca;
ax.XAxis.Exponent = 0;
lim2 = axis;
title({'Uniform Distribution'})
% Gaussian
ax3 = nexttile;
tile3 = area(elms, [scs_push_gauss, scs_rebuild_gauss, scs_migrate_gauss]);
ax = gca;
ax.XAxis.Exponent = 0;
lim3 = axis;
title({'Gaussian Distribution'})
% Exponential
ax4 = nexttile;
tile4 = area(elms, [scs_push_exp, scs_rebuild_exp, scs_migrate_exp]);
ax = gca;
ax.XAxis.Exponent = 0;
lim4 = axis;
title({'Exponential Distribution'})

lg = legend(nexttile(3), {'SCS pseudo-push'; 'SCS rebuild'; 'SCS migrate'});
lg.Location = 'northeastoutside';
% align axes
limits = [lim2; lim3; lim4];
limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits )

saveas(f,'smallE_largeP_AreaSCS.png')


% CabM Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'CabM Average Function Times')
xlabel(t, {'Number Particles (Ten Thousands)','Number Elements'})
ylabel(t, 'Seconds')
% Even (Excluded)
% Uniform
ax2 = nexttile;
area(elms, [cabm_push_uni, cabm_rebuild_uni, cabm_migrate_uni]);
ax = gca;
ax.XAxis.Exponent = 0;
lim2 = axis;
title({'Uniform Distribution'})
% Gaussian
ax3 = nexttile;
area(elms, [cabm_push_gauss, cabm_rebuild_gauss, cabm_migrate_gauss]);
ax = gca;
ax.XAxis.Exponent = 0;
lim3 = axis;
title({'Gaussian Distribution'})
% Exponential
ax4 = nexttile;
area(elms, [cabm_push_exp, cabm_rebuild_exp, cabm_migrate_exp]);
ax = gca;
ax.XAxis.Exponent = 0;
lim4 = axis;
title({'Exponential Distribution'})

lg = legend(nexttile(3), {'CabM pseudo-push'; 'CabM rebuild'; 'CabM migrate'});
lg.Location = 'northeastoutside';
% align axes
limits = [lim2; lim3; lim4];
limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits )

saveas(f,'smallE_largeP_AreaCabM.png')