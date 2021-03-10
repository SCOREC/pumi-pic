% MATLAB file for graph generation
clear

bytes = 268;
migrate = true;

%% Data Reading
fileID_rebuild = fopen( strcat('data/smallE_largeP_rebuild',int2str(bytes),'.dat'));
if not(migrate)
    fileID_push = fopen( strcat('data/smallE_largeP_push',int2str(bytes),'.dat'));
else
    fileID_push = fopen( strcat('data/smallE_largeP_migrate',int2str(bytes),'.dat'));
end

rebuild_data = fscanf(fileID_rebuild, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_push);

% element_number, particles_moved, average_time
elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0, 3:5 );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, 3:5 );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, 3:5 );
scs_push = push_data( push_data(:,1) == 0, 3:5 );
csr_push = push_data( push_data(:,1) == 1, 3:5 );
cabm_push = push_data( push_data(:,1) == 2, 3:5 );

%% Data Filtering
% {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% CabM (50%)
cabm_50 = cabm_rebuild( cabm_rebuild(:,2) == 50,:);
cabm_even_50 = cabm_50( cabm_50(:,1) == 0, 3);
cabm_uni_50 = cabm_50( cabm_50(:,1) == 1, 3);
cabm_gauss_50 = cabm_50( cabm_50(:,1) == 2, 3);
cabm_exp_50 = cabm_50( cabm_50(:,1) == 3, 3);
% CSR (50%)
csr_50 = csr_rebuild( csr_rebuild(:,2) == 50,:);
csr_even_50 = csr_50( csr_50(:,1) == 0, 3);
csr_uni_50 = csr_50( csr_50(:,1) == 1, 3);
csr_gauss_50 = csr_50( csr_50(:,1) == 2, 3);
csr_exp_50 = csr_50( csr_50(:,1) == 3, 3);
% SCS (50%)
scs_50 = scs_rebuild( scs_rebuild(:,2) == 50,:);
scs_even_50 = scs_50( scs_50(:,1) == 0, 3);
scs_uni_50 = scs_50( scs_50(:,1) == 1, 3);
scs_gauss_50 = scs_50( scs_50(:,1) == 2, 3);
scs_exp_50 = scs_50( scs_50(:,1) == 3, 3);

% CabM (Pseudo-Push/Migrate)
cabm_push = cabm_push( cabm_push(:,2) == 50,:);
cabm_even_push = cabm_push( cabm_push(:,1) == 0, 3);
cabm_uni_push = cabm_push( cabm_push(:,1) == 1, 3);
cabm_gauss_push = cabm_push( cabm_push(:,1) == 2, 3);
cabm_exp_push = cabm_push( cabm_push(:,1) == 3, 3);
% CSR (Pseudo-Push/Migrate)
csr_push = csr_push( csr_push(:,2) == 50,:);
csr_even_push = csr_push( csr_push(:,1) == 0, 3);
csr_uni_push = csr_push( csr_push(:,1) == 1, 3);
csr_gauss_push = csr_push( csr_push(:,1) == 2, 3);
csr_exp_push = csr_push( csr_push(:,1) == 3, 3);
% SCS (Pseudo-Push/Migrate)
scs_push = scs_push( scs_push(:,2) == 50,:);
scs_even_push = scs_push( scs_push(:,1) == 0, 3);
scs_uni_push = scs_push( scs_push(:,1) == 1, 3);
scs_gauss_push = scs_push( scs_push(:,1) == 2, 3);
scs_exp_push = scs_push( scs_push(:,1) == 3, 3);

%% Graph Generation
% Even
% figure(1)
% semilogy( ...
%     elms, scs_even_50./cabm_even_50, 'r--', ... % CabM Rebuild 50%
%     elms, scs_even_push./cabm_even_push, 'r:', ... % CabM Pseudo-Push
%     elms, scs_even_50./csr_even_50, 'b--', ... % CSR Rebuild 50%
%     elms, scs_even_push./csr_even_push, 'b:', ... % CSR Pseudo-Push
%     elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
% ax = gca;
% ax.XAxis.Exponent = 0;
% ax.XTick = 0:250:2000;
% ax.YTick = [1,10]; 
% ax.YTick = [0.1,1,10]; 
% ax.YTickLabel = {'0.1x','1x', '10x'}; 
% xlabel( {'Number Particles (Thousands)','Number Elements'} )
% ylabel("Structure Speedup (SCS/Structure)")
% legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
%      'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
%     'SCS (Reference)', 'Location', 'southeast')
% title({'Speedup (Even Distribution)','1:10,000 Element to Particle Ratio'})
% saveas(1,"smallE_largeP_even.png")

% Uniform
figure(2)
if not(migrate)
    semilogy( ...
        elms, scs_uni_50./cabm_uni_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_uni_push./cabm_uni_push, 'r:', ... % CabM Pseudo-Push
        elms, scs_uni_50./csr_uni_50, 'b--', ... % CSR Rebuild 50%
        elms, scs_uni_push./csr_uni_push, 'b:', ... % CSR Pseudo-Push
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
    legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
        'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
        'SCS (Reference)')
else
    semilogy( ...
        elms, scs_uni_50./cabm_uni_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_uni_push./cabm_uni_push, 'r-.', ... % CabM Pseudo-Push
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
    legend('CabM Rebuild 50%', 'CabM Migrate', ...
        'SCS (Reference)', ...
        'Location', 'southeast')
end
ax = gca;
ax.YTick = [0.1,1,10]; 
ax.YTickLabel = {'0.1x','1x', '10x'};
ax.XTick = 0:250:2000;
ax.YGrid = 'on';
xlabel( {'Number Particles (Ten Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
title({'Speedup (Uniform Distribution)','1:10,000 Element to Particle Ratio'})
saveas(2,"smallE_largeP_uniform.png")

% Gaussian
figure(3)
if not(migrate)
    semilogy( ...
        elms, scs_gauss_50./cabm_gauss_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_gauss_push./cabm_gauss_push, 'r:', ... % CabM Pseudo-Push
        elms, scs_gauss_50./csr_gauss_50, 'b--', ... % CSR Rebuild 50%
        elms, scs_gauss_push./csr_gauss_push, 'b:', ... % CSR Pseudo-Push
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
    legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
        'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
        'SCS (Reference)')
else
    semilogy( ...
        elms, scs_gauss_50./cabm_gauss_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_gauss_push./cabm_gauss_push, 'r-.', ... % CabM Migrate
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
    legend('CabM Rebuild 50%', 'CabM Migrate', ...
        'SCS (Reference)', ...
        'Location', 'southeast')
end
ax = gca;
ax.XTick = 0:250:2000;
ax.YTick = [0.1,1,10]; 
ax.YTickLabel = {'0.1x','1x', '10x'}; 
ax.YGrid = 'on';
xlabel( {'Number Particles (Ten Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
title({'Speedup (Gaussian Distribution)','1:10,000 Element to Particle Ratio'})
saveas(3,"smallE_largeP_gaussian.png")

% Exponential
figure(4)
if not(migrate)
    semilogy( ...
        elms, scs_exp_50./cabm_exp_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_exp_push./cabm_exp_push, 'r:', ... % CabM Pseudo-Push
        elms, scs_exp_50./csr_exp_50, 'b--', ... % CSR Rebuild 50%
        elms, scs_exp_push./csr_exp_push, 'b:', ... % CSR Pseudo-Push
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 )
    legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
        'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
        'SCS (Reference)')
else
    semilogy( ...
        elms, scs_exp_50./cabm_exp_50, 'r--', ... % CabM Rebuild 50%
        elms, scs_exp_push./cabm_exp_push, 'r-.', ... % CabM Migrate
        elms, ones(size(elms)), 'k', 'LineWidth', 0.75 )
    legend('CabM Rebuild 50%', 'CabM Migrate', ...
        'SCS (Reference)', ...
        'Location', 'southeast')
end
ax = gca;
ax.XTick = 0:250:2000;
ax.YTick = [0.1,1,10]; 
ax.YTickLabel = {'0.1x','1x', '10x'}; 
ax.YGrid = 'on';
xlabel( {'Number Particles (Ten Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
title({'Speedup (Exponential Distribution)','1:10,000 Element to Particle Ratio'})
saveas(4,"smallE_largeP_exponential.png")