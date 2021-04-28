% MATLAB file for graph generation
clear

% Excluded: CSR Migrate, Even Distribution

YTick = [0.01,0.1,0.5,1,5,10,100]; 
YTickLabel = {'0.01','0.1x','0.5x','1x','5x','10x','100x'};
LineWidth = 1.5;

%% Data Reading
fileID_rebuild = fopen('data/smallE_largeP_rebuild.dat');
fileID_push = fopen('data/smallE_largeP_push.dat');
fileID_migrate = fopen('data/smallE_largeP_migrate.dat');

% struct, element_number, distribution, average_time
rebuild_data = fscanf(fileID_rebuild, "%d %d %d %f", [4 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %f", [4 Inf])';
fclose(fileID_push);
migrate_data = fscanf(fileID_migrate, "%d %d %d %f", [4 Inf])';
fclose(fileID_migrate);

%% Data Filtering

% find length of graphs
elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
scs_length = length(unique(rebuild_data( rebuild_data(:,1) == 0, 2 )));
csr_length = length(unique(rebuild_data( rebuild_data(:,1) == 1, 2 )));
cabm_length = length(unique(rebuild_data( rebuild_data(:,1) == 2, 2 )));

% Only take instances with pull distribution and time
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0,[3,4] );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, [3,4] );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, [3,4] );
scs_push = push_data( push_data(:,1) == 0, [3,4] );
csr_push = push_data( push_data(:,1) == 1, [3,4] );
cabm_push = push_data( push_data(:,1) == 2, [3,4] );
scs_migrate = migrate_data( migrate_data(:,1) == 0, [3,4] );
csr_migrate = migrate_data( migrate_data(:,1) == 1, [3,4] );
cabm_migrate = migrate_data( migrate_data(:,1) == 2, [3,4] );

% Separate data by distribution, {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% SCS Rebuild
scs_rebuild_gauss = scs_rebuild( scs_rebuild(:,1) == 2, 2);
% CSR Rebuild
csr_rebuild_gauss = csr_rebuild( csr_rebuild(:,1) == 2, 2);
% CabM Rebuild
cabm_rebuild_gauss = cabm_rebuild( cabm_rebuild(:,1) == 2, 2);

% SCS Pseudo-Push
scs_push_gauss = scs_push( scs_push(:,1) == 2, 2);
% CSR Pseudo-Push
csr_push_gauss = csr_push( csr_push(:,1) == 2, 2);
% CabM Pseudo-Push
cabm_push_gauss = cabm_push( cabm_push(:,1) == 2, 2);

% SCS Migrate
scs_migrate_gauss = scs_migrate( scs_migrate(:,1) == 2, 2);
% % CSR Migrate
csr_migrate_gauss = csr_migrate( csr_migrate(:,1) == 2, 2);
% CabM Migrate
cabm_migrate_gauss = cabm_migrate( cabm_migrate(:,1) == 2, 2);

%% Graph Generation

% figure setup
f = figure;
f.Position(3:4) = [1100,350];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'Particle Structure Speedup (Gaussian Distribution)', 'FontWeight', 'bold')
xlabel(t, {'Number Particles (Ten Thousands)','Number Elements'}, 'FontWeight', 'bold')
ylabel(t, {'Average Structure Speedup','(SCS Time/Structure Time)'}, 'FontWeight', 'bold')

% Push
ax2 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_push_gauss(1:cabm_length)./cabm_push_gauss, 'r--', ....
    elms(1:csr_length), scs_push_gauss(1:csr_length)./csr_push_gauss, 'b--', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
lim2 = axis;
title({'Pseudo-Push'})

ax3 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_rebuild_gauss(1:cabm_length)./cabm_rebuild_gauss, 'r--', ...
    elms(1:csr_length), scs_rebuild_gauss(1:csr_length)./csr_rebuild_gauss, 'b--', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
lim3 = axis;
title({'Rebuild'})

ax4 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_migrate_gauss(1:cabm_length)./cabm_migrate_gauss, 'r--', ...
    elms(1:csr_length), scs_migrate_gauss(1:csr_length)./csr_migrate_gauss, 'b--', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
lim4 = axis;
title({'Migrate'})

lg = legend(nexttile(3), {'CabM'; 'CSR'}, 'FontWeight', 'bold');
lg.Location = 'northeastoutside';

% align axes
limits = [lim2; lim3; lim4];
%limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
limits = [ min(limits(:,1)), 5500, min(limits(:,3)), 5 ];
axis([ax2 ax3 ax4], limits )

saveas(f,'smallE_largeP_gauss.png')