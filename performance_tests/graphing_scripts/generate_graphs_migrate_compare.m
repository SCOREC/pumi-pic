clear

%% Data Reading
fileID_rebuild_before = fopen( 'data/rebuild/largeE_largeP_rebuild268.dat' );
fileID_rebuild_after = fopen( 'data/migrate/largeE_largeP_rebuild268.dat' );

rebuild_before_data = fscanf(fileID_rebuild_before, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild_before);
rebuild_after_data = fscanf(fileID_rebuild_after, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild_after);

% element_number, particles_moved, average_time
elms = unique(rebuild_before_data( rebuild_before_data(:,1) == 0, 2 ));
scs_before = rebuild_before_data( rebuild_before_data(:,1) == 0, 3:5 );
csr_before = rebuild_before_data( rebuild_before_data(:,1) == 1, 3:5 );
cabm_before = rebuild_before_data( rebuild_before_data(:,1) == 2, 3:5 );
scs_after = rebuild_after_data( rebuild_after_data(:,1) == 0, 3:5 );
csr_after = rebuild_after_data( rebuild_after_data(:,1) == 1, 3:5 );
cabm_after = rebuild_after_data( rebuild_after_data(:,1) == 2, 3:5 );

%% Data Filtering

% CabM Rebuild 50% - Before Migrate
cabm_before = cabm_before( cabm_before(:,2) == 50,:);
cabm_even_before = cabm_before( cabm_before(:,1) == 0, 3);
cabm_uni_before = cabm_before( cabm_before(:,1) == 1, 3);
cabm_gauss_before = cabm_before( cabm_before(:,1) == 2, 3);
cabm_exp_before = cabm_before( cabm_before(:,1) == 3, 3);
% CSR Rebuild 50% - Before Migrate
csr_before = csr_before( csr_before(:,2) == 50,:);
csr_even_before = csr_before( csr_before(:,1) == 0, 3);
csr_uni_before = csr_before( csr_before(:,1) == 1, 3);
csr_gauss_before = csr_before( csr_before(:,1) == 2, 3);
csr_exp_before = csr_before( csr_before(:,1) == 3, 3);
% SCS Rebuild 50% - Before Migrate
scs_before = scs_before( scs_before(:,2) == 50,:);
scs_even_before = scs_before( scs_before(:,1) == 0, 3);
scs_uni_before = scs_before( scs_before(:,1) == 1, 3);
scs_gauss_before = scs_before( scs_before(:,1) == 2, 3);
scs_exp_before = scs_before( scs_before(:,1) == 3, 3);

% CabM Rebuild 50% - After Migrate
cabm_after = cabm_after( cabm_after(:,2) == 50,:);
cabm_even_after = cabm_after( cabm_after(:,1) == 0, 3);
cabm_uni_after = cabm_after( cabm_after(:,1) == 1, 3);
cabm_gauss_after = cabm_after( cabm_after(:,1) == 2, 3);
cabm_exp_after = cabm_after( cabm_after(:,1) == 3, 3);
% CSR Rebuild 50% - After Migrate
csr_after = csr_after( csr_after(:,2) == 50,:);
csr_even_after = csr_after( csr_after(:,1) == 0, 3);
csr_uni_after = csr_after( csr_after(:,1) == 1, 3);
csr_gauss_after = csr_after( csr_after(:,1) == 2, 3);
csr_exp_after = csr_after( csr_after(:,1) == 3, 3);
% SCS Rebuild 50% - After Migrate
scs_after = scs_after( scs_after(:,2) == 50,:);
scs_even_after = scs_after( scs_after(:,1) == 0, 3);
scs_uni_after = scs_after( scs_after(:,1) == 1, 3);
scs_gauss_after = scs_after( scs_after(:,1) == 2, 3);
scs_exp_after = scs_after( scs_after(:,1) == 3, 3);


%% Graph Generation
% Even (Add in later?)

% Uniform
figure(2)
semilogx( elms, scs_uni_before, 'r--', ...
    elms, scs_uni_after(1:size(elms),:), 'r-.', ...
    elms, cabm_uni_before, 'b--', ...
    elms, cabm_uni_after(1:size(elms),:), 'b-.')
legend( 'SCS Rebuild (Without Migration)', 'SCS Rebuild (With Migration)', ...
    'CabM Rebuild (Without Migration)', 'CabM Rebuild (With Migration)', ...
    'Location', 'northwest')
ax = gca;
ax.YGrid = 'on';
ax.XTick = [1000,10000,100000]; 
ax.XTickLabel = {'1,000', '10,000', '100,000'};
xlabel( {'Number Elements','Number Particles (Thousands)'} )
ylabel("Average Time (Seconds)")
title({'Average Rebuild Time (Uniform Distribution)'})
saveas(2,"migrate_compare_uniform.png")

% Gaussian
figure(3)
semilogx( elms, scs_gauss_before, 'r--', ...
    elms, scs_gauss_after(1:size(elms),:), 'r-.', ...
    elms, cabm_gauss_before, 'b--', ...
    elms, cabm_gauss_after(1:size(elms),:), 'b-.')
legend( 'SCS Rebuild (Without Migration)', 'SCS Rebuild (With Migration)', ...
    'CabM Rebuild (Without Migration)', 'CabM Rebuild (With Migration)', ...
    'Location', 'northwest')
ax = gca;
ax.YGrid = 'on';
ax.XTick = [1000,10000,100000]; 
ax.XTickLabel = {'1,000', '10,000', '100,000'};
xlabel( {'Number Elements','Number Particles (Thousands)'} )
ylabel("Average Time (Seconds)")
title({'Average Rebuild Time (Gaussian Distribution)'})
saveas(3,"migrate_compare_gaussian.png")

% Exponential
figure(4)
semilogx( elms, scs_exp_before, 'r--', ...
    elms, scs_exp_after(1:size(elms),:), 'r-.', ...
    elms, cabm_exp_before, 'b--', ...
    elms, cabm_exp_after(1:size(elms),:), 'b-.')
legend( 'SCS Rebuild (Without Migration)', 'SCS Rebuild (With Migration)', ...
    'CabM Rebuild (Without Migration)', 'CabM Rebuild (With Migration)', ...
    'Location', 'northwest')
ax = gca;
ax.YGrid = 'on';
ax.XTick = [1000,10000,100000]; 
ax.XTickLabel = {'1,000', '10,000', '100,000'};
xlabel( {'Number Elements','Number Particles (Thousands)'} )
ylabel("Average Time (Seconds)")
title({'Average Rebuild Time (Exponential Distribution)'})
saveas(4,"migrate_compare_exponential.png")