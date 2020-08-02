close all
clear all
clc

load Yale_32x32

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% fea = NormalizeFea(fea);
% 
% ClassNum=length(unique(gnd));
% SampleNum=size(fea,1);
% SampleNumPerCalss=SampleNum/ClassNum;
% 
% X=[];
% 
% for Class=1:15
%     for Sample=1:11
%         X=[X fea(SampleNumPerCalss*(Class-1)+Sample,:)'];
%     end
% end
% K=Class;
% 
% [D N]=size(X);
% 
% save DataMatrix X D N K gnd 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load DataMatrix 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dist=zeros(N,N);
% for i=1:N
%     for j=1:N
%         Dist(i,j)=norm(X(:,i)-X(:,j),2);
%     end
% end
% 
% i=1;
% for const=[1/32 1/16 1/8 1/4 1/2 1 2 4 8 16 32]
%     KernelMatrices.K{i}=normpdf(Dist,0,const*max(max(Dist)));         
%     KernelMatrices.K{i}=KernelMatrices.K{i}/sum(sum(KernelMatrices.K{i}));
%     
%     figure;
%     mesh(KernelMatrices.K{i});
%     
%     i=i+1;
% end
% 
% KernelMatrices.MatrixNum=i-1;
% save KernelMatrices KernelMatrices Dist K N
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load KernelMatrices 

addpath('..\\CaiCode');
nClass = length(unique(gnd));
% rand('twister',5489);

load NMFParaInital
Options.H=Hinit;
Options.W=Winit;
Options.F=Finit;

Options.Method='fw';
Options.BalanPara=100;
Options.P=7;
Options.MaxIter=5;
Options.FeatureNum=256;
Options.GraphWeightType='Guass';
% Options.H=rand(D,K);
% Options.W=rand(K,N);
[H W Repre]=AdapGrNMF(X,[],Options);

save YaleAdapGrNMFfwResult H W Repre

label = litekmeans(W',nClass,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the AdapGrNMFfw subspace. MIhat: ',num2str(MIhat)]);
% Clustering in the AdapGrNMFfw subspace. MIhat: 0.41799

Options.Method='MultiK';
Options.K=K;
% Options.F=rand(N,K);
[H W Repre]=AdapGrNMF([],KernelMatrices,Options);

label = litekmeans(W',nClass,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the AdapGrNMFMultiK subspace. MIhat: ',num2str(MIhat)]);
% Clustering in the AdapGrNMFMultiK subspace. MIhat: 0.26895

Options.Method='MultiK';
Options.BalanPara=0;
[H W Repre]=AdapGrNMF([],KernelMatrices,Options);

label = litekmeans(W',nClass,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the NMFMultiK subspace. MIhat: ',num2str(MIhat)]);
% Clustering in the NMFMultiK subspace. MIhat: 0.24719

Options.Method='Cai';
[H W Repre]=AdapGrNMF(X,[],Options);

label = litekmeans(W',nClass,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the GrNMF subspace. MIhat: ',num2str(MIhat)]);
% Clustering in the GrNMF subspace. MIhat: 0.25672

Options.Method='Cai';
Options.BalanPara=0;
[H W Repre]=AdapGrNMF(X,[],Options);

save YaleGrNMFResult H W Repre

label = litekmeans(W',nClass,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the NMF subspace. MIhat: ',num2str(MIhat)]);
% Clustering in the NMF subspace. MIhat: 0.21775

% Clustering in the AdapGrNMFfw subspace. MIhat: 0.37868
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Clustering in the AdapGrNMFMultiK subspace. MIhat: 0.2948
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Optimization terminated.
% Clustering in the NMFMultiK subspace. MIhat: 0.24645
% Clustering in the GrNMF subspace. MIhat: 0.25615
% Clustering in the NMF subspace. MIhat: 0.24758

% Hinit=Options.H;
% Winit=Options.W;
% Finit=Options.F;
% 
% save NMFParaInital Hinit Winit Finit