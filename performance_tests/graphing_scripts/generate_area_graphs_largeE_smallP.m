% MATLAB file for graph generation
clear

% Excluded: Even Distribution

%% Data Reading
fileID_rebuild = fopen('data/largeE_smallP_rebuild.dat');
fileID_push = fopen('data/largeE_smallP_push.dat');
fileID_migrate = fopen('data/largeE_smallP_migrate.dat');

% struct, element_number, distribution, average_time
rebuild_data = fscanf(fileID_rebuild, "%d %d %d %f", [4 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %f", [4 Inf])';
fclose(fileID_push);
migrate_data = fscanf(fileID_migrate, "%d %d %d %f", [4 Inf])';
fclose(fileID_migrate);

%% Data Filtering

% find length of graphs
scs_length = length(unique(rebuild_data( rebuild_data(:,1) == 0, 2 )));
csr_length = length(unique(rebuild_data( rebuild_data(:,1) == 1, 2 )));
cabm_length = length(unique(rebuild_data( rebuild_data(:,1) == 2, 2 )));
dps_length = length(unique(rebuild_data( rebuild_data(:,1) == 3, 2 )));
if ( max( [scs_length, csr_length, cabm_length] ) == scs_length )
    elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
elseif ( max( [scs_length, csr_length, cabm_length] ) == csr_length )
    elms = unique(rebuild_data( rebuild_data(:,1) == 1, 2 ));
elseif ( max( [scs_length, csr_length, cabm_length] ) == dps_length )
    elms = unique(rebuild_data( rebuild_data(:,1) == 2, 2 ));
else
    elms = unique(rebuild_data( rebuild_data(:,1) == 3, 2 ));
end

% pull distributions and times
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0,[3,4] );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, [3,4] );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, [3,4] );
dps_rebuild = rebuild_data( rebuild_data(:,1) == 3, [3,4] );
scs_push = push_data( push_data(:,1) == 0, [3,4] );
csr_push = push_data( push_data(:,1) == 1, [3,4] );
cabm_push = push_data( push_data(:,1) == 2, [3,4] );
dps_push = push_data( push_data(:,1) == 3, [3,4] );
scs_migrate = migrate_data( migrate_data(:,1) == 0, [3,4] );
csr_migrate = migrate_data( migrate_data(:,1) == 1, [3,4] );
cabm_migrate = migrate_data( migrate_data(:,1) == 2, [3,4] );
dps_migrate = migrate_data( migrate_data(:,1) == 3, [3,4] );

% Separate data by distribution, {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% SCS Rebuild
%scs_rebuild_even = scs_rebuild( scs_rebuild(:,1) == 0, 2);
scs_rebuild_uni = scs_rebuild( scs_rebuild(:,1) == 1, 2);
scs_rebuild_gauss = scs_rebuild( scs_rebuild(:,1) == 2, 2);
scs_rebuild_exp = scs_rebuild( scs_rebuild(:,1) == 3, 2);
% CSR Rebuild
%csr_rebuild_even = csr_rebuild( csr_rebuild(:,1) == 0, 2);
csr_rebuild_uni = csr_rebuild( csr_rebuild(:,1) == 1, 2);
csr_rebuild_gauss = csr_rebuild( csr_rebuild(:,1) == 2, 2);
csr_rebuild_exp = csr_rebuild( csr_rebuild(:,1) == 3, 2);
% CabM Rebuild
%cabm_rebuild_even = cabm_rebuild( cabm_rebuild(:,1) == 0, 2);
cabm_rebuild_uni = cabm_rebuild( cabm_rebuild(:,1) == 1, 2);
cabm_rebuild_gauss = cabm_rebuild( cabm_rebuild(:,1) == 2, 2);
cabm_rebuild_exp = cabm_rebuild( cabm_rebuild(:,1) == 3, 2);
% DPS Rebuild
%dps_rebuild_even = dps_rebuild( dps_rebuild(:,1) == 0, 2);
dps_rebuild_uni = dps_rebuild( dps_rebuild(:,1) == 1, 2);
dps_rebuild_gauss = dps_rebuild( dps_rebuild(:,1) == 2, 2);
dps_rebuild_exp = dps_rebuild( dps_rebuild(:,1) == 3, 2);

% SCS Pseudo-Push
%scs_push_even = scs_push( scs_push(:,1) == 0, 2);
scs_push_uni = scs_push( scs_push(:,1) == 1, 2);
scs_push_gauss = scs_push( scs_push(:,1) == 2, 2);
scs_push_exp = scs_push( scs_push(:,1) == 3, 2);
% CSR Pseudo-Push
%csr_push_even = csr_push( csr_push(:,1) == 0, 2);
csr_push_uni = csr_push( csr_push(:,1) == 1, 2);
csr_push_gauss = csr_push( csr_push(:,1) == 2, 2);
csr_push_exp = csr_push( csr_push(:,1) == 3, 2);
% CabM Pseudo-Push
%cabm_push_even = cabm_push( cabm_push(:,1) == 0, 2);
cabm_push_uni = cabm_push( cabm_push(:,1) == 1, 2);
cabm_push_gauss = cabm_push( cabm_push(:,1) == 2, 2);
cabm_push_exp = cabm_push( cabm_push(:,1) == 3, 2);
% DPS Pseudo-Push
%dps_push_even = dps_push( dps_push(:,1) == 0, 2);
dps_push_uni = dps_push( dps_push(:,1) == 1, 2);
dps_push_gauss = dps_push( dps_push(:,1) == 2, 2);
dps_push_exp = dps_push( dps_push(:,1) == 3, 2);

% SCS Migrate
%scs_migrate_even = scs_migrate( scs_migrate(:,1) == 0, 2);
scs_migrate_uni = scs_migrate( scs_migrate(:,1) == 1, 2);
scs_migrate_gauss = scs_migrate( scs_migrate(:,1) == 2, 2);
scs_migrate_exp = scs_migrate( scs_migrate(:,1) == 3, 2);
% % CSR Migrate
%csr_migrate_even = csr_migrate( csr_migrate(:,1) == 0, 2);
csr_migrate_uni = csr_migrate( csr_migrate(:,1) == 1, 2);
csr_migrate_gauss = csr_migrate( csr_migrate(:,1) == 2, 2);
csr_migrate_exp = csr_migrate( csr_migrate(:,1) == 3, 2);
% CabM Migrate
%cabm_migrate_even = cabm_migrate( cabm_migrate(:,1) == 0, 2);
cabm_migrate_uni = cabm_migrate( cabm_migrate(:,1) == 1, 2);
cabm_migrate_gauss = cabm_migrate( cabm_migrate(:,1) == 2, 2);
cabm_migrate_exp = cabm_migrate( cabm_migrate(:,1) == 3, 2);
% DPS Migrate
%dps_migrate_even = dps_migrate( dps_migrate(:,1) == 0, 2);
dps_migrate_uni = dps_migrate( dps_migrate(:,1) == 1, 2);
dps_migrate_gauss = dps_migrate( dps_migrate(:,1) == 2, 2);
dps_migrate_exp = dps_migrate( dps_migrate(:,1) == 3, 2);

%% Graph Generation

% SCS Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'SCS Average Function Times')
xlabel(t, {'Number Particles (Thousands)','Number Elements'})
ylabel(t, 'Seconds')
% Even (Excluded)
% Uniform
ax2 = nexttile;
area(elms, [scs_push_uni, scs_rebuild_uni, scs_migrate_uni]);
ax = gca;
ax.XAxis.Exponent = 0;
lim2 = axis;
title({'Uniform Distribution'})
% Gaussian
ax3 = nexttile;
area(elms, [scs_push_gauss, scs_rebuild_gauss, scs_migrate_gauss]);
ax = gca;
ax.XAxis.Exponent = 0;
lim3 = axis;
title({'Gaussian Distribution'})
% Exponential
ax4 = nexttile;
area(elms, [scs_push_exp, scs_rebuild_exp, scs_migrate_exp]);
ax = gca;
ax.XAxis.Exponent = 0;
lim4 = axis;
title({'Exponential Distribution'})

lg = legend(nexttile(3), {'SCS pseudo-push'; 'SCS rebuild'; 'SCS migrate'});
lg.Location = 'northeastoutside';
% align axes
limits = [lim2; lim3; lim4];
%limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
limits = [ min(limits(:,1)), 55000, min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits)

saveas(f,'largeE_smallP_AreaSCS.png')


% CSR Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'CSR Average Function Times')
xlabel(t, {'Number Particles (Thousands)','Number Elements'})
ylabel(t, 'Seconds')
% Even (Excluded)
% Uniform
ax2 = nexttile;
area(elms, [csr_push_uni, csr_rebuild_uni, csr_migrate_uni]);
ax = gca;
ax.XAxis.Exponent = 0;
lim2 = axis;
title({'Uniform Distribution'})
% Gaussian
ax3 = nexttile;
area(elms, [csr_push_gauss, csr_rebuild_gauss, csr_migrate_gauss]);
ax = gca;
ax.XAxis.Exponent = 0;
lim3 = axis;
title({'Gaussian Distribution'})
% Exponential
ax4 = nexttile;
area(elms, [csr_push_exp, csr_rebuild_exp, csr_migrate_exp]);
ax = gca;
ax.XAxis.Exponent = 0;
lim4 = axis;
title({'Exponential Distribution'})

lg = legend(nexttile(3), {'CSR pseudo-push'; 'CSR rebuild'; 'CSR migrate'});
lg.Location = 'northeastoutside';
% align axes
limits = [lim2; lim3; lim4];
%limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
limits = [ min(limits(:,1)), 55000, min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits)

saveas(f,'largeE_smallP_AreaCSR.png')


% CabM Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'CabM Average Function Times')
xlabel(t, {'Number Particles (Thousands)','Number Elements'})
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
%limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
limits = [ min(limits(:,1)), 55000, min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits)

saveas(f,'largeE_smallP_AreaCabM.png')


% DPS Graphs
% figure setup
f = figure;
f.Position(3:4) = [1000,300];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'CabM Average Function Times')
xlabel(t, {'Number Particles (Thousands)','Number Elements'})
ylabel(t, 'Seconds')
% Even (Excluded)
% Uniform
ax2 = nexttile;
area(elms, [dps_push_uni, dps_rebuild_uni, dps_migrate_uni]);
ax = gca;
ax.XAxis.Exponent = 0;
lim2 = axis;
title({'Uniform Distribution'})
% Gaussian
ax3 = nexttile;
area(elms, [dps_push_gauss, dps_rebuild_gauss, dps_migrate_gauss]);
ax = gca;
ax.XAxis.Exponent = 0;
lim3 = axis;
title({'Gaussian Distribution'})
% Exponential
ax4 = nexttile;
area(elms, [dps_push_exp, dps_rebuild_exp, dps_migrate_exp]);
ax = gca;
ax.XAxis.Exponent = 0;
lim4 = axis;
title({'Exponential Distribution'})

lg = legend(nexttile(3), {'DPS pseudo-push'; 'DPS rebuild'; 'DPS migrate'});
lg.Location = 'northeastoutside';
% align axes
limits = [lim2; lim3; lim4];
%limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
limits = [ min(limits(:,1)), 55000, min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits)

saveas(f,'largeE_smallP_AreaDPS.png')