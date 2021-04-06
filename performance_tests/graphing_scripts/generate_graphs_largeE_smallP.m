% MATLAB file for graph generation
clear

% Excluded: CSR Migrate, Even Distribution

%% Data Reading
fileID_rebuild = fopen('data/largeE_smallP_rebuild.dat');
fileID_push = fopen('data/largeE_smallP_push.dat');
fileID_migrate = fopen('data/largeE_smallP_migrate.dat');

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
csr_length = length(unique(rebuild_data( rebuild_data(:,1) == 1, 2 )));
cabm_length = length(unique(rebuild_data( rebuild_data(:,1) == 2, 2 )));

% Only take instances with 50% particles moved, pull distribution and time
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0,[3,5] );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, [3,5] );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, [3,5] );
scs_push = push_data( push_data(:,1) == 0, [3,5] );
csr_push = push_data( push_data(:,1) == 1, [3,5] );
cabm_push = push_data( push_data(:,1) == 2, [3,5] );
scs_migrate = migrate_data( migrate_data(:,1) == 0, [3,5] );
csr_migrate = migrate_data( migrate_data(:,1) == 1, [3,5] );
cabm_migrate = migrate_data( migrate_data(:,1) == 2, [3,5] );

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

%% Graph Generation

% figure setup
f = figure;
f.Position(3:4) = [1100,350];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'Particle Structure Speedup')
xlabel(t, {'Number Particles (Thousands)','Number Elements'})
ylabel(t, {'Average Structure Speedup','(SCS Time/Structure Time)'})

% Even (excluded)


% Uniform
ax2 = nexttile;
loglog( ...
    elms(1:cabm_length), scs_push_uni(1:cabm_length)./cabm_push_uni, 'r:', ...
    elms(1:cabm_length), scs_rebuild_uni(1:cabm_length)./cabm_rebuild_uni, 'r--', ...
    elms(1:cabm_length), scs_migrate_uni(1:cabm_length)./cabm_migrate_uni, 'r-.', ...
    elms(1:csr_length), scs_push_uni(1:csr_length)./csr_push_uni, 'b:', ...
    elms(1:csr_length), scs_rebuild_uni(1:csr_length)./csr_rebuild_uni, 'b--', ...
    elms(1:csr_length), scs_migrate_uni(1:csr_length)./csr_migrate_uni, 'b-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', 0.75 );
ax = gca;
ax.XTick = [100, 1000, 10000];
ax.XTickLabel = {'100', '1000', '10000'};
ax.YTick = [0.01,0.1,1,10,100]; 
ax.YTickLabel = {'0.01','0.1x','1x', '10x', '100x'};
ax.YGrid = 'on';
title({'Uniform Distribution'})
lim2 = axis;

% Gaussian
ax3 = nexttile;
loglog( ...
    elms(1:cabm_length), scs_push_gauss(1:cabm_length)./cabm_push_gauss, 'r:', ...
    elms(1:cabm_length), scs_rebuild_gauss(1:cabm_length)./cabm_rebuild_gauss, 'r--', ...
    elms(1:cabm_length), scs_migrate_gauss(1:cabm_length)./cabm_migrate_gauss, 'r-.', ...
    elms(1:csr_length), scs_push_gauss(1:csr_length)./csr_push_gauss, 'b:', ...
    elms(1:csr_length), scs_rebuild_gauss(1:csr_length)./csr_rebuild_gauss, 'b--', ...
    elms(1:csr_length), scs_migrate_gauss(1:csr_length)./csr_migrate_gauss, 'b-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', 0.75 );
ax = gca;
ax.XTick = [100, 1000, 10000];
ax.XTickLabel = {'100', '1000', '10000'};
ax.YTick = [0.01,0.1,1,10,100]; 
ax.YTickLabel = {'0.01','0.1x','1x', '10x', '100x'};
ax.YGrid = 'on';
title({'Gaussian Distribution'})
lim3 = axis;

% Exponential
ax4 = nexttile;
loglog( ...
    elms(1:cabm_length), scs_push_exp(1:cabm_length)./cabm_push_exp, 'r:', ...
    elms(1:cabm_length), scs_rebuild_exp(1:cabm_length)./cabm_rebuild_exp, 'r--', ...
    elms(1:cabm_length), scs_migrate_exp(1:cabm_length)./cabm_migrate_exp, 'r-.', ...
    elms(1:csr_length), scs_push_exp(1:csr_length)./csr_push_exp, 'b:', ...
    elms(1:csr_length), scs_rebuild_exp(1:csr_length)./csr_rebuild_exp, 'b--', ...
    elms(1:csr_length), scs_migrate_exp(1:csr_length)./csr_migrate_exp, 'b-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', 0.75 );
ax = gca;
ax.XTick = [100, 1000, 10000];
ax.XTickLabel = {'100', '1000', '10000'};
ax.YTick = [0.01,0.1,1,10,100]; 
ax.YTickLabel = {'0.01x','0.1x','1x', '10x', '100x'};
ax.YGrid = 'on';
title({'Exponential Distribution'})
lim4 = axis;

lg = legend(nexttile(3), {'CabM pseudo-push'; 'CabM rebuild'; 'CabM migrate'; ...
                          'CSR pseudo-push'; 'CSR rebuild'; 'CSR migrate'; ...
                          'SCS (reference)'});
lg.Location = 'northeastoutside';

% align axes
limits = [lim2; lim3; lim4];
limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits )

saveas(f,'largeE_smallP.png')