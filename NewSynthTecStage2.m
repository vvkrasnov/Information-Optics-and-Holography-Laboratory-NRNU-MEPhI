clear
close all
vTime0=clock;
disp('Program is running...')

grad=2;        % levels
MinDif=0.000001; %Minimum difference
T0=0;           %Annealing temperature
MaxIter=5;     %Maximum number of iterations     

alpha=0.5;  %0 - only DE, 1 - only NSTD

Name='1_256';    %Original image
KName='GS_D_F0.3_1_2561024x1024K_GS_F_D_NSTD0.34531_DE0.059148BI38Sp0';    %Start hologram

dH=10.8*10^(-6);    %Hologram pixel size
l=633*10^(-9);      %Wavelength
MaxNiter = 50;      %Number ofiterations to perform
FocusLense=0.3;     %Distanse between point sourse and hologram;

GraphRes=100;      %Graphics resolution: number of dots for one full iteration
RTI=1;              %Toggle for graphics plotting 
LogScale=0;


Named=strcat(Name,'.tif'); 
KNamed=strcat(KName,'.bmp');
SaveName=strcat(Name,'MinDif',num2str(MinDif),'An',num2str(T0));

RandStream.setGlobalStream (RandStream('mt19937ar','seed',1));
I2=double(imread(Named));

I=I2.^0.5;

Bit=log2(grad);

Temp=double(imread(KNamed));
Temp=double(Temp/(max(Temp(:))));
C=gpuArray(Temp);

XY(1)=floor((size(Temp,1)-size(I,1))/2)+1;
XY(2)=floor((size(Temp,2)-size(I,2))/2)+1;

Crop=[XY(2) XY(1) size(I2,2)-1 size(I2,1)-1];

XSize=size(C,2);
YSize=size(C,1);

A= ones(size(C),'gpuArray');

FN=XSize*YSize;
Dif=1;
vI=0;
vBmin=inf;
n=1;

scrsz = get(groot,'ScreenSize');
             
N1=size(Temp,1);
N2=size(Temp,2);
N1_Rec=N1-1;
N2_Rec=N2-1;             

[a0,b0] = meshgrid(-N2_Rec/2 : N2_Rec/2, -N1_Rec/2 : N1_Rec/2);
a0=gpuArray(a0);
b0=gpuArray(b0);

LightPhase=angle(double(exp(((-pi*1i/(FocusLense*l)).*(dH*dH).*(a0.*a0 + b0.*b0))))); 
LightAmplitude = ones(size(Temp));
for i=1:N1
    for j=1:N2
        LightAmplitude(i,j)=(FocusLense^2/(FocusLense^2+(dH^2)*((i-N1/2)^2+(j-N2/2)^2))).^0.5;
    end
end
LightAmplitude=gpuArray(LightAmplitude);

K=LightAmplitude.*C.*exp(1i.*LightPhase);
Pg=ifftshift(ifft2(fftshift(K)));
Pg=abs(Pg.*conj(Pg));
P=gather(Pg);

[NSTD,DE]=NSTDandDE(P,I2,Crop);
DE=DE*mean(C(:));
vBL=alpha*NSTD+(1-alpha)*(1-DE);
DEmin=DE;
NSTDmin=NSTD;

disp(strcat('NSTD=',num2str(NSTD),'_DE=',num2str(DE),'_TF=',num2str(vBL),'_Best TF=',num2str(vBL)))

Name=strcat('DSRT_kinoform_etr_an_',num2str(T0),Name,'NSTD',num2str(vBL));

Num=0;
if RTI==1
    figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2])
    plot(Num,vBL,'rs','MarkerFaceColor','r','MarkerSize',3)   
    hold on
    plot(Num,NSTDmin,'bd','MarkerFaceColor','b','MarkerSize',3)
    plot(Num,DEmin,'g^','MarkerFaceColor','g','MarkerSize',3)
    drawnow    
    if LogScale==1
        set(gca,'XScale','log','XLim',[0.00001 MaxIter*2])
    end
    grid on
    xlabel('Iteration number')
    ylabel('Target Function')
    legend('Target function', 'NSTD','Diffraction losses')
end

vB=zeros(3,1);
NSTDc=zeros(3,1);
DEc=zeros(3,1);
PN=1;
while Dif>MinDif && vI<MaxIter
    Time=inf;
    vRij=randperm(XSize*YSize);
    vI=vI+1;
    T=T0/vI;
    for i4=1:FN
        Graph_dis=log(i4+(vI-1)*FN);    
        i=ceil(vRij(i4)/XSize);
        i2=vRij(i4)-(i-1)*XSize;
        Cii2=C(i,i2);
        j(1)=1;
        j(3)=2^Bit;
        for l=1:Bit
                j(2)=j(1)+floor((j(3)-j(1))/2);
                if l==1
                    for k=1:3
                        C(i,i2)=j(k)-1;
                        K=LightAmplitude.*C.*exp(1i.*LightPhase);
                        Pg=ifftshift(ifft2(fftshift(K)));
                        Pg=abs(Pg.*conj(Pg));
                        P=gather(Pg);
                        
                        [NSTD,DE]=NSTDandDE(P,I2,Crop);
                        DE=DE*gather(mean(C(:)));
                        vB(k)=alpha*NSTD+(1-alpha)*(1-DE);  
                        NSTDc(k)=NSTD;
                        DEc(k)=DE;
                        
                        if j(1)==j(3)
                            vB(2)=vB(1);
                            vB(3)=vB(1);
                            break
                        end
                    end
                    [vBjmax, n3]=max(vB);
                    [vBjmin, n1]=min(vB);
                    DEmin=DEc(n1);
                    NSTDmin=NSTDc(n1);

                    vBm=min(vB);
                    Cm=j(n1)-1;

                    vB(n1)=inf;
                    [vBjav, n2]=min(vB);
                    vB(n1)=vBjmin;
                    if abs(n1-n2)==2
                        n2=n3;
                        vT=vB(3);
                        vB(3)=vB(2);
                        vB(2)=vT;
                    end
                    if n1==1 || n2==1
                        j(3)=j(2);
                        vB(3)=vB(2);
                    elseif n1==3 || n2==3
                        j(1)=j(2);
                        vB(1)=vB(2);
                    end
                    if j(3)-j(1)==1                     
                        break
                    end 
                else                   
                    C(i,i2)=j(2)-1;
                    K=LightAmplitude.*C.*exp(1i.*LightPhase);
                    Pg=ifftshift(ifft2(fftshift(K)));
                    Pg=abs(Pg.*conj(Pg));
                    P=gather(Pg);

                    [NSTD,DE]=NSTDandDE(P,I2,Crop);
                    DE=DE*gather(mean(C(:)));
                    vB(2)=alpha*NSTD+(1-alpha)*(1-DE);
                    NSTDc(2)=NSTD;
                    DEc(2)=DE;
                    if vB(2)<vBm
                        vBm=vB(2);
                        Cm=j(2)-1;
                    end
                    [vBjmax, n3]=max(vB);
                    [vBjmin, n1]=min(vB);
                    DEmin=DEc(n1);
                    NSTDmin=NSTDc(n1);                    
                    vB(n1)=inf;
                    [vBjav, n2]=min(vB);
                    vB(n1)=vBjmin;
                    if abs(n1-n2)==2
                        n2=n3;
                        vT=vB(3);
                        vB(3)=vB(2);
                        vB(2)=vT;
                    end                        
                    if l~=Bit && j(3)-j(1)>1
                        if n1==1 || n2==1
                            j(3)=j(2);
                            vB(3)=vB(2);
                        elseif n1==3 || n2==3
                            j(1)=j(2);
                            vB(1)=vB(2);
                        end  
                    end
                    if j(3)-j(1)==1
                        break
                    end                    
                end             
        end
        C(i,i2)=Cm;
        if vBjmin>vBmin
            Pr=exp((vBmin-vBjmin)/T);
            true=Pr-rand;
            if true<0
                C(i,i2)=Cii2;   
            else
                vBmin=vBjmin;
                DEmin=DEc(n1);
                NSTDmin=NSTDc(n1);             
            end
        else
            vBmin=vBjmin;
            DEmin=DEc(n1);
            NSTDmin=NSTDc(n1);           
        end      
        if rem(i4,100)==1
            disp(strcat('Iteration_',num2str(vI),',_',num2str(i4),'_Pixel_',num2str(i),'_',num2str(i2),', TF=',num2str(vBmin),'. NSTD=',num2str(NSTD),'. DE=',num2str(DE)))  
        end
        if RTI==1
            if (round(Graph_dis*GraphRes)>=PN && LogScale==1) || (rem(i4,round(FN/GraphRes))==1 && LogScale==0)
                PN=PN+1;
                figure (1)
                plot((vI-1)+i4/FN,vBmin,'rs','MarkerFaceColor','r','MarkerSize',3)   
                hold on
                plot((vI-1)+i4/FN,NSTDmin,'bd','MarkerFaceColor','b','MarkerSize',3)
                plot((vI-1)+i4/FN,DEmin,'g^','MarkerFaceColor','g','MarkerSize',3)
                legend('Target function', 'NSTD','Diffraction efficiency')
                drawnow  
            end
        end
    end
    K=LightAmplitude.*C.*exp(1i.*LightPhase);
    Pg=ifftshift(ifft2(fftshift(K)));
    Pg=abs(Pg.*conj(Pg));
    P=gather(Pg);

    [NSTD,DE]=NSTDandDE(P,I2,Crop);
    DE=DE*gather(mean(C(:)));
    vB=alpha*NSTD+(1-alpha)*(1-DE);    
    
    disp(strcat('TF=',num2str(vB),'. NSTD=',num2str(NSTD),'. DE=',num2str(DE)))
    Dif=(vBL-vB)/vBL;
    vBL=vB;
    Name1=strcat('DSRT_',KName,'_','NSTD',num2str(NSTD),'_DE',num2str(DE));
    C2=gather(C);
    % Saving to file
    imwrite(uint8(C2*255),strcat(Name1,'.bmp'),'bmp')    %Hologram
    imwrite(uint8((P/max(P(:)))*255),strcat(Name1,'_psf','.tif'),'tif') %Reconstruction plane
end

load handel.mat;
sound(y/2);

vTime1=clock;
vTimeWorked=vTime1-vTime0;
vTotalSeconds=vTimeWorked(3)*86400+vTimeWorked(4)*3600+vTimeWorked(5)*60+vTimeWorked(6);
vHours=floor(vTotalSeconds/3600);
vMinutes=floor((vTotalSeconds-3600*vHours)/60);
vSeconds=vTotalSeconds-3600*vHours-60*vMinutes;
disp(strcat('All_finished_in_',num2str(vHours),'_hours_',num2str(vMinutes),'_minutes_',num2str(vSeconds),'_seconds.'))