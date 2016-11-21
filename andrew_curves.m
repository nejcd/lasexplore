clc;
clear all;

    
ds = dataset('file','E:\Dropbox\dev\Data\29.csv','Delimiter',',','ReadObsNames',false);

ds.Properties.VarNames{1} = 'linearity';
ds.Properties.VarNames{2} = 'planarity';
ds.Properties.VarNames{3} = 'scattering';
ds.Properties.VarNames{4} = 'omnivariance';
ds.Properties.VarNames{5} = 'anisotropy';
ds.Properties.VarNames{6} = 'eigentropy';
ds.Properties.VarNames{7} = 'classification';
ds.Properties.VarNames{8} = 'intensity';



%All classification
feat = [ds.linearity, ds.planarity, ds.scattering,...
    ds.omnivariance, ds.anisotropy, ds.eigentropy];
class = ds.classification;
figure(1)
title('Andrew curwes all class')
andrewsplot(feat,'group',class,'quantile',.25)

%Ground and buliding
dsGroundBuilding = ds(ds.classification==2 | ds.classification==6,:);
featGroundBuilding = [dsGroundBuilding.linearity,...
    dsGroundBuilding.planarity, dsGroundBuilding.scattering,...
    dsGroundBuilding.omnivariance, dsGroundBuilding.anisotropy,...
    dsGroundBuilding.eigentropy];
class = dsGroundBuilding.classification;
figure(2)
title('Andrew curwes - ground & building')
andrewsplot(featGroundBuilding,'group',class,'quantile',.25)