function [H W Repre]=AdapGrNMF(DataMatrix,KernelMatrices,Options)
% [H W Repre]=AdapGrNMF(DataMatrix,KernelMatrices,Options)
%  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% load DataMatrix
% 
% Options.Method='fw';
% Options.BalanPara=100;
% Options.P=7;
% Options.MaxIter=5;
% Options.FeatureNum=256;
% Options.GraphWeightType='Guass';
% Options.H=rand(D,K);
% Options.W=rand(K,N);
% [H W Repre]=AdapGrNMF(X,[],Options);
% 
% figure;
% subplot(1,2,1)
% pcolor([W; W(K,:)]);title('W');
% 
% ImgFeaWeight=reshape(sum(Repre.Lambda),32,32);
% subplot(1,2,2)
% pcolor(ImgFeaWeight);
% title('Feature weights')
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% load KernelMatrices
% 
% Options.Method='MultiK';
% Options.BalanPara=1;
% Options.P=7;
% Options.MaxIter=5;
% Options.GraphWeightType='Guass';
% Options.K=K;
% Options.F=rand(N,K);
% Options.W=rand(K,N);
% [H W Repre]=AdapGrNMF([],KernelMatrices,Options);
% F=H;
% figure;
% subplot(1,2,1)
% pcolor([F F(:,K)]);title('F')
% subplot(1,2,2)
% pcolor([W; W(K,:)]);title('W')
% figure;
% bar(Repre.tau);title('Kernel weights')
% 

alpha=Options.BalanPara;
P=Options.P;
T=Options.MaxIter;

if strcmp(Options.Method,'Cai')
    SampleNum=size(DataMatrix,2);
    Dim=size(DataMatrix,1);
    
    lambda=ones(Dim,1);%*Options.FeatureNum/Dim;
    
    Lambda=diag(lambda);
    
    X=DataMatrix;
    H=Options.H;
    W=Options.W;
    
    fea=(Lambda*DataMatrix)';
    Dist = EuDist2(fea,fea);
    [Y,I]=sort(Dist,2,'ascend');
    Neighboors=I(:,2:P+1);
    A=zeros(SampleNum,SampleNum);

    for n=1:SampleNum
        if strcmp(Options.GraphWeightType,'0-1')
            A(n,Neighboors(n,:))=1;
        else
            A(n,Neighboors(n,:))=normpdf(Dist(n,Neighboors(n,:)),0,mean(mean(Dist)));
        end
    end    
    D=diag(sum(A,1));
    
    for t=1:Options.MaxIter    
        
        %%%%%%Update the factorization matrices Ht and Wt as in (22);

        H=(Lambda*Lambda*X*W')./(Lambda*Lambda*H*W*W').*H;
        
        W=(H'*Lambda*Lambda*X+Options.BalanPara*W*A)./...
            (H'*Lambda*Lambda*H*W+Options.BalanPara*W*D).*W;            
    end
    
    Repre.Type='Cai';
    Repre.Lambda=Lambda;            

end

if strcmp(Options.Method,'fw') 
    SampleNum=size(DataMatrix,2);
    Dim=size(DataMatrix,1);
    
    lambda=ones(Dim,1)*Options.FeatureNum/Dim;
    
    Lambda=diag(lambda);
    
    X=DataMatrix;
    H=Options.H;
    W=Options.W;
    
    for t=1:Options.MaxIter    
        
        %%%%%%%%%Update the graph Gt and its corresponding Laplacian matrix Lt according to t?1 = diag(?t?1) as
        %%%%%%%%% introduce in section 3.1.2;
        fea=(Lambda*DataMatrix)';
        Dist = EuDist2(fea,fea);
        [Y,I]=sort(Dist,2,'ascend');
        Neighboors=I(:,2:P+1);
        A=zeros(SampleNum,SampleNum);
        
        for n=1:SampleNum
            if strcmp(Options.GraphWeightType,'0-1')
                A(n,Neighboors(n,:))=1;
            else
                A(n,Neighboors(n,:))=normpdf(Dist(n,Neighboors(n,:)),0,mean(mean(Dist)));
            end
        end    
        D=diag(sum(A,1));
        
        %%%%%%Update the factorization matrices Ht and Wt as in (22);

        H=(Lambda*Lambda*X*W')./(Lambda*Lambda*H*W*W').*H;
        
        W=(H'*Lambda*Lambda*X+Options.BalanPara*W*A)./...
            (H'*Lambda*Lambda*H*W+Options.BalanPara*W*D).*W;

        
        %%%%%Update the feature weights ?t as in (25);
        C=Options.FeatureNum;
        Y=X-H*W;
        e=zeros(Dim,1);
        for d=1:Dim
            e(d)=Y(d,:)*Y(d,:)';
        end
        
        Hqp=diag(e);
        f=zeros(Dim,1);
        A=eye(Dim);
        b=ones(Dim,1)*C;
        Aeq=ones(Dim,1)';
        beq=C;
        lb=zeros(Dim,1);
        ub=ones(Dim,1)*C;  
        
        options = optimset('Display','off','Algorithm','interior-point-convex');        
        
        lambda = quadprog(Hqp,f,A,b,Aeq,beq,lb,ub,[],options);
        Lambda=diag(lambda);         
            
    end
    
    Repre.Type='fw';
    Repre.Lambda=Lambda;            
end

if strcmp(Options.Method,'MultiK') 
    K=Options.K;
    F=Options.F;
    W=Options.W;

    L=KernelMatrices.MatrixNum;

    Kernel=zeros(size(KernelMatrices.K{1}));
    tau=ones(L,1)/L;
    N=size(Kernel,1);

    for t=1:T
        Kernel=zeros(N,N);
        for i=1:L
            Kernel=Kernel+tau(i)*KernelMatrices.K{i};
        end    

        for n=1:N
            for m=1:N
                DistK(n,m)=Kernel(n,n)+Kernel(m,m)-2*Kernel(n,m);
            end
        end
        [Y,I] = sort(DistK,2,'ascend');
        Nn=I(:,2:(P+1));

        A=zeros(N,N);
        for n=1:N
            if strcmp(Options.GraphWeightType,'0-1')
                A(n,Nn(n))=1;
            else
                A(n,Nn(n))=Kernel(n,Nn(n));
            end
        end

        D=diag(sum(A,1));

        F=F.*(Kernel*W')./(Kernel*F*W*W');
        W=W.*(F'*Kernel+alpha*W*A)./(F'*K*F*W+alpha*W*D);

        Z=eye(N)-F*W;
        for i=1:L
            g(i,1)=trace(KernelMatrices.K{i}*Z*Z');
        end
        
        lb=zeros(L,1);
        ub=ones(L,1);
        
        tau = linprog(g,-eye(L),zeros(L,1),ones(1,L),1,lb,ub);
    end
    
    H=F;
    W=W;
    Repre.Type='MultiK';
    Repre.tau=tau;    
end


function D = EuDist2(fea_a,fea_b,bSqrt)

if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end

