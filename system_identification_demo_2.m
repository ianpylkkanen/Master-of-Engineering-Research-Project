function system_identification_demo

%--------------------------------------------------------------------------
%------ Matlab Code for the Identification of Stochastic Systems ----------
%--------------------------------------------------------------------------
%----------------------  by Fotis P. Kopsaftopoulos -----------------------
%--------------------------------------------------------------------------
% Last Update: 10 - 04 - 2020 by F. Kopsaftopoulos
%
% This demo requires a MATLAB version 7 (R2007a) or higher. 
% Toolboxes required: system identification, signal processing, 
% statistics and control systems.
%
% INSTRUCTIONS
%
% 1. Create a .mat file with the output and input signals, if available.
% The vector or matrix with the output signal(s) should be named as "y"
% with N x M dimensions, N designating the number of data samples and M the
% number of output (response) signals. Similarly the vector or matrix with the
% input signal(s) should be named as "x" with N x L dimensions, N designating  
% the number of data samples and L the number of input (excitation)
% signals.
% 
% 2. Place the .mat file in the same folder as the
% system_identification_demo.
%
% 3. Type in the command window "system_identification_demo".
%
% 4. Provide the name of the .mat file that contains the data. Do NOT add
% the .mat extension.
%
% 5. Follow the guidelines in the command window.
%
% 6. After a function is performed you need to press any key for the code
% to continue.
%
%
% RUNNING THE CODE - COPYING/PASTING PARTS
% All figures may be also obtained after executing the 'Data' and 
% 'Normalized signals' parts and running each figure separetely. 

%--------------------------------------------------------------------------
%                                 Data 
%--------------------------------------------------------------------------

clc

datafile=input('Give the file containing the excitation/response (input/output) data: \n','s');

load(datafile)

fs=input('Give the sampling frequency (fs): \n');

N=size(y,1);
outputs=size(y,2);

if exist('x','var')==1
    nx=size(x,2);
else
    x=[]; nx=size(x,2);   
end

if isempty(fs)
    fs = 1; % (Hz)
end

TT=0:1/fs:(length(y)-1)/fs;

%--------------------------------------------------------------------------

i=10; ii=14; % Font sizes --> select according to your preferences

%--------------------------------------------------------------------------
%-------------------------- Acquired signals ------------------------------
%--------------------------------------------------------------------------

if nx==1
    figure
    plot(TT,x)
    set(gca,'fontsize',i),box on
    title('Excitation Signal (Input)','Fontsize',ii)
    xlabel('Time (s)','Fontsize',ii)
    ylabel('Force (N)','Fontsize',ii)
    xlim([0 TT(end)])
end

%pause

%i=10; ii=14; % Font sizes --> select according to your preferences

figure
for output=1:outputs
    subplot(outputs,1,output),plot(TT,y(:,output))
    set(gca,'fontsize',i),box on
    feval('title',sprintf('Output: %d',output),'fontsize',ii)
    ylabel('Acceleration (m/s^2)','Fontsize',i)
    xlabel('Time (s)','Fontsize',i)
    xlim([0 TT(end)])
end

pause  % ,close


%--------------------------------------------------------------------------
%                          Normalized signals 
%--------------------------------------------------------------------------

ans1=input('Do you want to detrend/normalize the data?   (yes or no)  \n','s');

if strcmp(ans1,'yes')
    
    Y = detrend(y);
    for output=1:outputs
        %Y(:,output)=Y(:,output)./max(abs(Y(:,output))); 
        Y(:,output)=Y(:,output)./std(Y(:,output)); 
    end
    
    if nx==1
        X = detrend(x);
        %X = X./max(abs(X));
        X = X./std(X);
    else
        X = detrend(x);
    end
    
    i=10; ii=14;
    
    if nx==1
    figure
    plot(TT,X)
    set(gca,'fontsize',i),box on
    title('Normalized Excitation Signal','Fontsize',20)
    xlabel('Time (s)','Fontsize',ii)
    ylabel('Force (Nt)','Fontsize',ii)
    xlim([0 TT(end)])
    end

    figure
    for output=1:outputs
        subplot(outputs,1,output),plot(TT,Y(:,output))
        set(gca,'fontsize',i),box on
        feval('title',sprintf('Output: %d',output),'fontsize',ii)
        ylabel('Normalized Response Signal','Fontsize',i)
        xlabel('Time (s)','Fontsize',i)
        xlim([0 TT(end)])
    end

elseif  nx==1; X=x; Y=y; 
else Y=y; X=[];
end


%--------------------------------------------------------------------------
%                    Histogram and Normal Probability Plot
%--------------------------------------------------------------------------

i=10; ii=14; 

for output=1:outputs
    figure
    subplot(2,1,1),histfit(Y(:,output),round(sqrt(N)))
    set(gca,'fontsize',i),box on
    feval('title',sprintf('Sensor 6'),'fontsize',ii)
    subplot(2,1,2), normplot(Y(:,output))
    set(gca,'fontsize',i),box on
end

pause


%--------------------------------------------------------------------------
%                                  ACFs 
%--------------------------------------------------------------------------

if nx==1
    
    figure
    acf_wn(X,100,0.8);
    ylim([-1 1])
    title('Excitation (input) AutoCorrelation Function (ACF)','fontsize',ii)
%     [acf_x,lags,bounds_acf]=autocorr(X,100); 
%     figure
%     bar(acf_x,0.5)
%     line([0 size(acf_x,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
%     line([0 size(acf_x,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
%     axis([0 100 -1 1])
%     set(gca,'fontsize',i)
%     title('Excitation (input) AutoCorrelation Function (ACF)','fontsize',ii)
%     ylabel('Excitation ACF','fontsize',ii,'interpreter','tex')
%     xlabel('Lag','fontsize',ii,'interpreter','tex')
end

%[ccf_xy,lags,bounds_ccf]=crosscorr(X,Y,100);

i=10; ii=14;

figure
for output=1:outputs
    
    subplot(outputs,1,output),acf_wn(Y(:,output),100,0.8);
    ylim([-1 1])
    set(gca,'fontsize',i)
    feval('title',sprintf('Response ACF (output %d)',output),'fontsize',ii)
    
%     [acf_y(:,output),lags,bounds_acf]=autocorr(Y(:,output),100);
%     subplot(outputs,1,output),bar(acf_y(:,output),0.5)
%     line([0 size(acf_y,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
%     line([0 size(acf_y,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
%     axis([0 100 -1 1])
%     set(gca,'fontsize',i)
%     feval('title',sprintf('Response ACF (output %d)',output),'fontsize',ii)
%     ylabel('Response ACF','fontsize',ii,'interpreter','tex')
%     xlabel('Lag','fontsize',ii,'interpreter','tex')
end

pause

clc

%--------------------------------------------------------------------------
%--------------- Deterministic excitation-response spectra ----------------
%--------------------------------------------------------------------------

disp('-----------------------------------');
disp('      Non-parametric analysis      ')
disp('-----------------------------------');
        
disp('Deterministic (FFT) based response spectra')
WINDOW=input('Give the FFT points (samples): \n');

if isempty(x)
    
    figure
    for output=1:outputs
        %Sy(:,output)=fft(Y(:,output),WINDOW); ws=linspace(0,128,size(Sy(:,output),1)/2)'; 
        Sy(:,output)=fft(Y(:,output),WINDOW); ws=linspace(0,fs/2,size(Sy(:,output),1)/2)'; 
        Syy(:,output)=Sy(:,output).*conj(Sy(:,output)); 
        H{output}=Syy(:,output); % Ho=Sy./Sx; %figure,plot(w,20*log10(abs(H(1:size(H,1)/2,1))))
        %phh=unwrap(angle(H)); Phh=(phh.*180)./pi;
        subplot(outputs,1,output),plot(ws,20*log10(abs(H{output}(1:size(H{output},1)/2,1))))
        set(gca,'fontsize',i),box on
        xlim([0 fs/2])
        feval('title',sprintf('Deterministic spectrum (FFT based) for output %d',output),'fontsize',ii)
        ylabel('PSD (dB)','Fontsize',ii)
        xlabel('Frequency (Hz)','Fontsize',ii)
    end
    
else
    figure
    for output=1:outputs
        %Sy(:,output)=fft(Y(:,output),1026); Sx=fft(X,1026); w=linspace(0,128,size(Sy(:,output),1)/2)'; 
        Sy(:,output)=fft(Y(:,output),WINDOW); Sx=fft(X,WINDOW); w=linspace(0,fs/2,size(Sy(:,output),1)/2)'; 
        Syy(:,output)=Sy(:,output).*conj(Sy(:,output)); Sxx=Sx.*conj(Sx);
        Sxy(:,output)=Sx.*conj(Sy(:,output));
        H{output}=Sxy(:,output)./Sxx; % Ho=Sy./Sx; %figure,plot(w,20*log10(abs(H(1:size(H,1)/2,1))))
        %phh=unwrap(angle(H)); Phh=(phh.*180)./pi;
        subplot(outputs,1,output),plot(w,20*log10(abs(H{output}(1:size(H{output},1)/2,1))))
        set(gca,'fontsize',i),box on
        xlim([0 fs/2])
        feval('title',sprintf('Deterministic FRF (FFT based) for output %d',output),...
            'fontsize',ii)
        ylabel('PSD (dB)','Fontsize',ii)
        xlabel('Frequency (Hz)','Fontsize',ii)
    end
end
  
pause

%--------------------------------------------------------------------------
%---------------- Welch based excitation-response spectra -----------------
%--------------------------------------------------------------------------

clc

disp('Welch based excitation-response spectra')
WINDOW=input('Give window length (samples): \n');
OVERLAP=input('Give window overlap percentage (a number from 0 to 1): \n');

NFFT=WINDOW;

%----------------------- estimation parameters ----------------------------
%--------------------------------------------------------------------------
if isempty(WINDOW); WINDOW = 1024; NFFT = WINDOW; end    
if isempty(OVERLAP); OVERLAP = 0.8; end                  
%--------------------------------------------------------------------------

%disp('-----------------------------------');
%disp('      Welch based estimation       ');
%disp('-----------------------------------');
%disp('Window        NFFT      Overlap (%)');
%disp('-----------------------------------');
%disp([ WINDOW; NFFT; OVERLAP]');

%pause

i=10; ii=14;

if nx==1
    [Pxx,w] = pwelch(X,WINDOW,round(OVERLAP*WINDOW),NFFT,fs); % excitation spectrum

    figure
    plot(w,20*log10(abs(Pxx)))
    set(gca,'fontsize',i),box on
    xlim([0 fs/2])
    %axis([5,80,-140,-40])
    title('Welch based excitation spectrum','Fontsize',ii)
    ylabel('PSD (dB)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',ii)
end

figure
for output=1:outputs
    [Pyy(:,output),w, Pyyc] = pwelch(Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs); % response spectrum
    subplot(outputs,1,output),plot(w,20*log10(abs(Pyy(:,output)))); hold on
    plot(w, 20*log10(Pyyc(:,1)), '-.r')
    plot(w, 20*log10(Pyyc(:,2)), '-.r')
    set(gca,'fontsize',i),box on
    xlim([0 fs/2])
    feval('title',sprintf('Welch based response spectrum for output %d',output),'fontsize',ii)
    ylabel('PSD (dB)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',ii)
    legend({'PSD estimate', '$\pm$ 95 \%'},'Orientation','horizontal','Location','northeast', 'interpreter', 'latex'); legend boxoff
end

pause % ,close


%--------------------------------------------------------------------------
%----------------  Excitation-response spectra comparison -----------------
%--------------------------------------------------------------------------

df=fs/WINDOW;

DATA_spa=iddata(Y,[],1/fs);
G_oo=spa(DATA_spa,WINDOW,0:df*2*pi:fs/2*2*pi); 
Goo=reshape(G_oo.spectrumdata,[outputs^2 size(G_oo.spectrumdata,3)])';

i=10; ii=14;

index=1:outputs+1:outputs^2;

figure
for output=1:outputs
    subplot(outputs,1,output), hold on
    plot(w,20*log10(abs(Pyy(:,output))))
    plot(G_oo.frequency/(2*pi),20*log10(abs(Goo(:,index(output)))),'r')
    %plot(ws,20*log10(abs(H{output}(1:size(H{output},1)/2,1))),'color',[0 0.5 0])
    set(gca,'fontsize',i), box on
    xlim([1 fs/2])
    feval('title',sprintf('Welch - Blackman-Tukey response spectral estimates comparison for output %d',output),...
        'Fontname','TimesNewRoman','fontsize',ii)
    ylabel('Spectrum (dB)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',i)
    legend('Welch based','Blackman-Tukey','Location','SouthEast','Orientation','horizontal')
end

pause


%--------------------------------------------------------------------------
%---------------- Welch based output spectra (overlap effect) -------------
%--------------------------------------------------------------------------

if outputs>1
    output=input('Select output to demonstrate overlap/window effect: \n');
else output=1;
end

%--------------------------------------------------------------------------
OVERLAP1 = 0.9; OVERLAP2 = 0.8; OVERLAP3 = 0.7; OVERLAP4 = 0.6; %           <----------
%--------------------------------------------------------------------------

[Pyy1,w] = pwelch(Y(:,output),WINDOW,round(OVERLAP1*WINDOW),NFFT,fs);
[Pyy2,w] = pwelch(Y(:,output),WINDOW,round(OVERLAP2*WINDOW),NFFT,fs);
[Pyy3,w] = pwelch(Y(:,output),WINDOW,round(OVERLAP3*WINDOW),NFFT,fs);
[Pyy4,w] = pwelch(Y(:,output),WINDOW,round(OVERLAP4*WINDOW),NFFT,fs);

i=12; ii=16;

figure, hold on
plot(w,20*log10(abs(Pyy1)),'k');
plot(w,20*log10(abs(Pyy2)),'b');
plot(w,20*log10(abs(Pyy3)),'r');
plot(w,20*log10(abs(Pyy4)),'g');
set(gca,'fontsize',i)
xlim([0 fs/2])
title('Welch based output spectrum (overlap effect)','Fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
legend('90 %','80 %','70 %','60 %','Location','SouthEast','Orientation','horizontal')

pause % ,close

%--------------------------------------------------------------------------
%---------------- Welch based output spectra (window effect) --------------
%--------------------------------------------------------------------------

clear Pyy* w

%--------------------------------------------------------------------------
%WINDOW1=256; WINDOW2=512; WINDOW3=1024; WINDOW4=2048; OVERLAP=0.75;
WINDOW1=128; WINDOW2=256; WINDOW3=512; WINDOW4=1024; OVERLAP=0.75;
%--------------------------------------------------------------------------

[Pyy1,w1] = pwelch(Y(:,output),WINDOW1,round(OVERLAP*WINDOW1),WINDOW1,fs);
[Pyy2,w2] = pwelch(Y(:,output),WINDOW2,round(OVERLAP*WINDOW2),WINDOW2,fs);
[Pyy3,w3] = pwelch(Y(:,output),WINDOW3,round(OVERLAP*WINDOW3),WINDOW3,fs);
[Pyy4,w4] = pwelch(Y(:,output),WINDOW4,round(OVERLAP*WINDOW4),WINDOW4,fs);

i=10; ii=14;

figure
plot(w1,20*log10(abs(Pyy1)),'k'),hold on
plot(w2,20*log10(abs(Pyy2)),'b')
plot(w3,20*log10(abs(Pyy3)),'r')
plot(w4,20*log10(abs(Pyy4)),':g')
set(gca,'fontsize',i)
xlim([0 fs/2])
title('Welch based output spectrum (window effect)','Fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
legend('128', '256','512','1024','Location','SouthEast','Orientation','horizontal')

pause % ,close


%--------------------------------------------------------------------------
%-------------------------- Coherence function ----------------------------
%--------------------------------------------------------------------------

if nx==1;
    
    for output=1:outputs
        [Cxy(:,output),w] = mscohere(X,Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs); % coherence estimation
    end
    
    i=10; ii=14;
    
    figure
    for output=1:outputs
        subplot(outputs,1,output),plot(w,Cxy(:,output))
        set(gca,'fontsize',i),box on
        xlim([0 fs/2])
        feval('title',sprintf('Coherence Function for output %d',output),'fontsize',ii)
        ylabel('Coherence','Fontsize',ii)
        xlabel('Frequency (Hz)','Fontsize',ii)
    end
    
    pause % ,close
    
% %--------------------------------------------------------------------------
% %------------ Blackman-Tukey based excitation-response spectra ------------
% %--------------------------------------------------------------------------
% 
% % G_o=spa(dat_o,1024,0:0.25*2*pi:80*2*pi);
% % figure,plot(G_o.frequency/(2*pi),20*log10(abs(reshape(G_o.spectrumdata,[1601 1]))))
% % G_io=spa(dat_io,1024,0:0.25*2*pi:80*2*pi);
% % figure,plot(G_io.frequency/(2*pi),20*log10(abs(reshape(G_io.spectrumdata,[1601 1]))))
% G_oo=spa(dat_oo,1024,0:0.25*2*pi:80*2*pi); Goo=reshape(G_oo.spectrumdata,[4 size(G_oo.spectrumdata,3)])';
% % figure,plot(G_oo.frequency/(2*pi),20*log10(abs(Goo(:,1))))
% ph=unwrap(angle(Goo(:,2))); % ph=angle(Goo(:,2)); ph=reshape(ph,[1601 1]);
% Ph=(ph.*180)./pi;
% 
% figure
% set(gcf,'PaperOrientation','landscape','papertype','A4',...
%     'paperunits','centimeters','paperposition',[0.63 0.63 28.41 19.72])
% subplot(3,1,1),plot(G_oo.frequency/(2*pi),20*log10(abs(Goo(:,4))))
% set(gca,'fontsize',16)
% title('Blackman-Tukey based spectral estimates','fontsize',20,'interpreter','tex')
% ylabel('PSD (dB)','fontsize',20,'interpreter','tex')
% annotation('textbox','String',{'(a)'},'FontSize',18,'FontName','Times New Roman',...
%     'FitHeightToText','off','LineStyle','none','Position',[0.8614 0.8844 0.02747 0.04397]);
% axis([4 80 -140 -40])
% subplot(3,1,2),plot(G_oo.frequency/(2*pi),20*log10(abs(Goo(:,2))))
% set(gca,'fontsize',16)
% ylabel('CSD magn. (dB)','fontsize',20,'interpreter','tex')
% axis([4 80 -140 -40])
% annotation('textbox','String',{'(b)'},'FontSize',18,'FontName','Times New Roman',...
%     'FitHeightToText','off','LineStyle','none','Position',[0.8592 0.5855 0.02747 0.04397]);
% subplot(3,1,3),plot(G_oo.frequency/(2*pi),-Ph)
% set(gca,'fontsize',16)
% ylabel('CSD phase (deg)','fontsize',20,'interpreter','tex')
% xlabel('Frequency (Hz)','fontsize',20,'interpreter','tex')
% axis([4 80 -400 250])
% annotation('textbox','String',{'(c)'},'FontSize',18,'FontName','Times New Roman',...
%     'FitHeightToText','off','LineStyle','none','Position',[0.86 0.2852 0.02747 0.04397]);
% 
% pause

%--------------------------------------------------------------------------
%---------------------------- Welch based FRF -----------------------------
%--------------------------------------------------------------------------

    clc

    disp('Welch based Frequency Response Function (FRF) estimates')
    WINDOW=input('Give window length (in samples) or press enter for default value: \n');
    OVERLAP=input('Give window overlap percentage (a number from 0 to 1) or press enter for default value: \n');

    NFFT=WINDOW;

%----------------------- estimation parameters ----------------------------
%--------------------------------------------------------------------------
    if isempty(WINDOW); WINDOW = 1024; NFFT = WINDOW; end    
    if isempty(OVERLAP); OVERLAP = 0.8; end                  
%--------------------------------------------------------------------------

    for output=1:outputs
        [Txy(:,output),w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs); % FRF estimation
        ph(:,output)=unwrap(angle(Txy(:,output))); Ph(:,output)=(ph(:,output).*180)./pi;
    end

% disp(' ');
% disp('      Welch based FRF estimation       ');
% disp('-----------------------------------');
% disp('Window        NFFT      Overlap (%)');
% disp('-----------------------------------');
% disp([ WINDOW; NFFT; OVERLAP]');

    i=12; ii=16;

    figure
    for output=1:outputs
        subplot(outputs,1,output),plot(w,20*log10(abs(Txy(:,output))))
        set(gca,'fontsize',i),box on
        xlim([0 fs/2])
        feval('title',sprintf('Welch based FRF magnitude for output %d',output),...
            'fontsize',ii)
        ylabel('Magnitude (dB)','Fontsize',ii)
        xlabel('Frequency (Hz)','Fontsize',ii)
    end

    pause % ,close

%--------------------------------------------------------------------------
%--------------------- Welch based FRF (overlap effect) -------------------
%--------------------------------------------------------------------------

    if outputs>1
        output=input('Select output to demonstrate overlap effect: \n');
    else output=1;
    end

%--------------------------------------------------------------------------
    OVERLAP1 = 0.9; OVERLAP2 = 0.8; OVERLAP3 = 0.7; OVERLAP4 = 0.6; %      <----------
%--------------------------------------------------------------------------

    [Txy1,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP1*WINDOW),NFFT,fs); 
    ph1=unwrap(angle(Txy1)); Ph1=(ph1.*180)./pi;
    [Txy2,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP2*WINDOW),NFFT,fs); 
    ph2=unwrap(angle(Txy2)); Ph2=(ph2.*180)./pi;
    [Txy3,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP3*WINDOW),NFFT,fs); 
    ph3=unwrap(angle(Txy3)); Ph3=(ph3.*180)./pi;
    [Txy4,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP4*WINDOW),NFFT,fs); 
    ph4=unwrap(angle(Txy4)); Ph4=(ph4.*180)./pi;

    i=10; ii=14;

    figure
    subplot(2,1,1),plot(w,20*log10(abs(Txy1)),'r'),hold on
    subplot(2,1,1),plot(w,20*log10(abs(Txy2)),'--','Color',[0.2 0.5 0.2])
    subplot(2,1,1),plot(w,20*log10(abs(Txy3)),'-.b')
    subplot(2,1,1),plot(w,20*log10(abs(Txy4)),':k')
    set(gca,'fontsize',i)
    xlim([0 fs/2])
    title('Welch based FRF (overlap effect)','Fontsize',ii)
    ylabel('Magnitude (dB)','Fontsize',ii)
    legend('90 %','80 %','70 %','60 %','Location','SouthEast','Orientation','horizontal')
    subplot(2,1,2),plot(w,Ph1,'r'),hold on
    subplot(2,1,2),plot(w,Ph2,'--','Color',[0.2 0.5 0.2])
    subplot(2,1,2),plot(w,Ph3,'-.b')
    subplot(2,1,2),plot(w,Ph4,':k')
    xlim([0 fs/2])
    set(gca,'fontsize',i)
    ylabel('Phase (deg)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',ii)
    legend('90 %','80 %','70 %','60 %','Location','SouthEast','Orientation','horizontal')

    pause % ,close

end


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%                           ARX Identification
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

clc

models=[];

if isempty(models)
    
    if isempty(x)
                
        if outputs>1
            output=input('Select output for which to run ARMA models:');
        else
            output=1;
        end
    
       
        DATA=iddata(Y(:,output),[],1/fs);
    
        minar=input('Give minimum AutoRegressive (AR) order: \n');
        maxar=input('Give maximum AutoRegressive (AR) order: \n');
        
%         disp('-----------------------------------');
%         disp('        ARMA Identification        ')
%         disp('-----------------------------------');

%         tic
%         for order=minar:maxar
%             models{order}=armax(DATA,[order order]);
%             modal{order}=the_modals(models{order}.c,models{order}.a,fs,1,1);
%         end
%         toc
                
        disp('-----------------------------------');
        disp('        AR Identification        ')
        disp('-----------------------------------');
        
        tic
        for order=minar:maxar
            models{order}=arx(DATA,order);
            modal{order}=the_modals(1,models{order}.a,fs,1,1);
        end
        toc
        
    else 
        disp('-----------------------------------');
        disp('         ARX Identification        ')
        disp('-----------------------------------');
    
        if outputs>1
            output=input('Select output for which to run ARX models:');
        else
            output=1;
        end
    
        DATA=iddata(Y(:,output),X,1/fs);
    
        minar=input('Give minimum AutoRegressive (AR) order: \n');
        maxar=input('Give maximum AutoRegressive (AR) order: \n');
    
        tic
        for order=minar:maxar
            models{order}=arx(DATA,[order order+1 0]);
        end
        toc
        
                    
    end

end


Yp=cell(1,maxar); rss=zeros(1,maxar); BIC=zeros(1,maxar);

for order=minar:maxar
    BIC(order)=log(models{order}.noisevariance)+...
         (size(models{order}.parametervector,1)*log(N))/N;
end

for order=minar:maxar
    Yp{order}=predict(models{order},DATA,1);
    rss(order)=100*(norm(DATA.outputdata-Yp{order}.outputdata)^2)/(norm(DATA.outputdata)^2);
end
    
%--------------------------------------------------------------------------
%-----------------------------  BIC-RSS plot ------------------------------
%--------------------------------------------------------------------------

i=10; ii=14;

figure
subplot(2,1,1),plot(minar:maxar,BIC(minar:maxar),'-o')
xlim([minar maxar])
title('BIC criterion','Fontsize',ii)
ylabel('BIC','Fontsize',ii)
set(gca,'fontsize',i)
subplot(2,1,2),plot(minar:maxar,rss(minar:maxar),'-o')
xlim([minar maxar])
title('RSS/SSS criterion','Fontsize',ii)
ylabel('RSS/SSS (%)','Fontsize',ii)
xlabel('AR(n)','Fontsize',ii)
set(gca,'fontsize',i)

pause 

%--------------------------------------------------------------------------
%----------------- ARX/ARMA frequency stabilization plot ------------------
%--------------------------------------------------------------------------

if isempty(x)
    
    [D,fn,z] = deal(zeros(maxar,round(maxar/2+1)));
    
    for order=minar:maxar
        qq=size(modal{order},1);
        D(order,1:qq) = modal{order}(:,3).';
        fn(order,1:qq) = modal{order}(:,1).';
        z(order,1:qq) = modal{order}(:,2).';
    end

else
    [D,fn,z] = deal(zeros(maxar,maxar/2));

    for order=minar:maxar
        clear num den
        num = models{order}.B;
        den = models{order}.A;
        [DELTA,Wn,ZETA,R,lambda]=disper_new(num,den,fs);
        qq = length(DELTA);
        D(order,1:qq) = DELTA';
        fn(order,1:qq) = Wn';
        z(order,1:qq) = ZETA';
    end

end



i=10; ii=14;

%s = 0.0005; % scaling factor
s = 10;

figure, hold on
for order=minar:maxar
    for jj=1:maxar/2
        %imagesc(fn(order,jj), order, z(order,jj));
        imagesc(s*fn(order,jj), order, z(order,jj))        
        %image(s*fn(order,jj), order, z(order,jj))
    end
end
%axis([0,5*fs/2,minar,maxar])
axis([1,s*fs/2,minar,maxar])
colorbar,box on,
h = get(gca,'xtick');
set(gca,'xticklabel',h/s,'fontsize',i);
%set(gca,'xticklabel',h,'fontsize',i);
title('Frequency stabilization plot (colormap indicates damping ratio)','Fontsize',ii)
if isempty(x)
    ylabel('AR(n)','Fontsize',ii)
else
    ylabel('ARX(n,n)','Fontsize',ii)
end
xlabel('Frequency (Hz)','Fontsize',ii)
%xlim([0 fs/2])

%text(5*fs/2,45,'Damping Ratio','Fontsize',ii,...
%    'Rotation',-90,'VerticalAlignment','Middle','HorizontalAlignment','center')

pause % ,close

%--------------------------------------------------------------------------
order=input('Select final model order: \n'); % select ARX/ARMA model orders <----------

disp('Natural Frequencies (Hz)');
disp(nonzeros(fn(order,:)))
 
disp('Damping Ratios (%)');
disp(nonzeros(z(order,:)))

pause

%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%----------------------------- ARX/ARMA FRF -------------------------------
%--------------------------------------------------------------------------

df=input('Select frequency resolution for parametric spectra: \n'); % 

if isempty(x)
    [MAG,PHASE,wp] = dbode(models{order}.c,models{order}.a,1/fs,2*pi*[0:df:fs/2]);
else
    [MAG,PHASE,wp] = dbode(models{order}.B,models{order}.A,1/fs,2*pi*[0:df:fs/2]);
end

i=10; ii=14;

figure
plot(wp/(2*pi),20*log10(abs(MAG)))
%plot(w,20*log10(abs(Txy)),'r')
xlim([0 fs/2])
set(gca,'fontsize',i)
if isempty(x)
    feval('title',sprintf('Parametric FRF for selected orders - ARMA(%d,%d)',order,order),...
        'Fontname','TimesNewRoman','fontsize',ii)
else
    feval('title',sprintf('Parametric FRF for selected orders - ARX(%d,%d)',order,order),...
        'Fontname','TimesNewRoman','fontsize',ii)
end
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)

pause % ,close

%--------------------------------------------------------------------------
%---------------- Parametric vs non-parametric FRF comparison -------------
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%order = 30; % select ARX orders                                           <----------
%WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; % Welch based parameters       <----------
%--------------------------------------------------------------------------

if isempty(x)
    [Pyy,w] = pwelch(Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs);
    [MAG,PHASE,wp] = ffplot(models{order},0:df:fs/2);
    MAG=reshape(MAG,[1 size(MAG,3)]);
    
    i=10; ii=14;

    figure
    plot(wp,20*log10(abs(MAG))),hold on
    plot(w,20*log10(abs(Pyy)),'r')
    xlim([0 fs/2])
    set(gca,'fontsize',i)
    title('Parametric (ARMA based) vs non-parametric (Welch based) FRF comparison',...
        'Fontname','TimesNewRoman','Fontsize',ii)
    ylabel('Magnitude (dB)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',ii)
    legend('Parametric','Welch based','Location','SouthEast','Orientation','vertical')

    pause 
    
else
    
    [Txy,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs);
    [MAG,PHASE,wp] = dbode(models{order}.B,models{order}.A,1/fs,2*pi*[0:df:fs/2]);

    i=10; ii=14;

    figure
    plot(wp/(2*pi),20*log10(MAG)),hold on
    plot(w,20*log10(abs(Txy)),'r')
    xlim([0 fs/2])
    set(gca,'fontsize',i)
    title('Parametric (ARX based) vs non-parametric (Welch based) FRF comparison',...
        'Fontname','TimesNewRoman','Fontsize',ii)
    ylabel('Magnitude (dB)','Fontsize',ii)
    xlabel('Frequency (Hz)','Fontsize',ii)
    legend('Parametric','Welch based','Location','SouthEast','Orientation','vertical')
end

pause 

%--------------------------------------------------------------------------
%--------------------- Parametric + confidence intervals ------------------
%--------------------------------------------------------------------------

[Magh,phh,wp,sdmagh,sdphaseh]=bode(models{order},[0:df:fs/2]*2*pi);
wp=wp./(2*pi);

Magh=reshape(Magh,[size(Magh,3) 1]); 
sdmagh=reshape(sdmagh,[size(sdmagh,3) 1]); sdphaseh=reshape(sdphaseh,[size(sdphaseh,3) 1]);

%colA=[0 1 0];
%colB=[1 0 0];
colC=[0.8 0.8 0.8];

i=10; ii=14; 

a=2; % standard deviations

figure
set(gcf,'paperorientation','landscape','paperposition',[0.63 0.63 28.41 19.72]);
%subplot(3,2,1) 
box on, hold on
plot(wp,20*log10(Magh),'color',colC); 
plot(wp,20*log10(Magh-a*sdmagh),'color',colC);
plot(wp,20*log10(Magh+a*sdmagh),'color',colC);
%plot(w,20*log10(abs(Txy)),'--b')
xlim([3 fs/2])
set(gca,'fontsize',i)
title('Parametric FRF with 95% confidence intervals','fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
%legend('Healhty','Damage A','Damage B','Damage C','Damage D','Damage E','Location','SouthEast',...
%    'Orientation','horizontal')
%legend([ph,pA,pB],'healthy','damage I','damage II','Location','SouthWest','Orientation','horizontal')
set(patch,'EdgeAlpha',0,'FaceAlpha',0)
patch([wp;flipud(wp)],[20*log10(abs(Magh-a*sdmagh));flipud(20*log10(Magh+a*sdmagh))],colC)

pause


%--------------------------------------------------------------------------
%----------------------------- Residual ACF -------------------------------
%--------------------------------------------------------------------------

res=DATA.outputdata-Yp{order}.outputdata;

figure
acf_wn(res(order+1:end),100,0.8);
title('Residuals ACF','fontsize',12)
ylim([-0.5 0.5])

% [acf_res,lags,bounds_acf]=autocorr(res(order+1:end),100); 
% figure
% bar(acf_res,0.8)
% line([0 size(acf_res,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
% line([0 size(acf_res,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
% axis([0 100 -0.5 0.5])
% set(gca,'fontsize',i)
% title('Model residuals AutoCorrelation Function (ACF)','fontsize',ii)
% ylabel('Excitation ACF','fontsize',ii,'interpreter','tex')
% xlabel('Lag','fontsize',ii,'interpreter','tex')
    
pause

%--------------------------------------------------------------------------
%                    Histogram and Normal Probability Plot
%--------------------------------------------------------------------------

N=length(res);

i=10; ii=14; 

figure
subplot(2,1,1),histfit(res,round(sqrt(N)))
set(gca,'fontsize',i),box on
title('Residuals','fontsize',ii)
subplot(2,1,2), normplot(res)
set(gca,'fontsize',i),box on

pause


%--------------------------------------------------------------------------
%                              Poles - Zeros
%--------------------------------------------------------------------------

figure
pzmap(models{order})

pause

%--------------------------------------------------------------------------
%                           Signal - predictions
%--------------------------------------------------------------------------

figure
plot(TT,DATA.outputdata,'-o'), hold on
plot(TT,Yp{order}.outputdata,'*')
title('Model one-step-ahead prediction (*) vs actual signal (o)','fontsize',12)
ylabel('Signal','Fontsize',ii)
xlabel('Time (s)','Fontsize',ii)
%xlim([11 11.3])

%--------------------------------------------------------------------------
% %------------------------------------------------------------------------
% %                           Auxiliary Functions
% %------------------------------------------------------------------------
% %------------------------------------------------------------------------


function [Delta,fn,z,R,lambda]=disper_new(num,den,Fs)

% num		: The numerator of the transfer function
% den		: The denominator of the transfer function
% Fs		: The sampling frequency (Hz)
% Delta	: The precentage dispersion
% fn		: The corresponding frequencies (Hz)
% z		: The corresponding damping (%)
% R		: The residues of the discrete system
% Mag		: The magnitude of the corresponding poles
% This function computes the dispersion of each frequency of a system. The System is  
% enetred as a transfer function. In case the order of numerator polynomial is greater than 
% that of the denominator the polynomial division is apllied, and the dispersion is considered at
% the remaine tf. The analysis is done using the Residuez routine of MATLAB.
% The results are printed in the screen in asceding order of frequencies.
% This routine displays only the dispersions from the natural frequencies (Complex Poles).

% REFERENCE[1]:  MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES - A Critical Assessment 
% REFERENCE[2]:  PANDIT WU

%--------------------------------------------
% Created	: 08 December 1999.
% Author(s)	: A. Florakis & K.A.Petsounis
% Updated	: 16 February 1999.
%--------------------------------------------

% Sampling Period
Ts=1/Fs;

% Calculate the residues of the Transfer Function
num=num(:).';
den=den(:).';

%---------------------------------------------------
% For Analysis with the contant term
%[UPOLOIPO,PILIKO]=deconv(fliplr(num),fliplr(den));
%UPOLOIPO=fliplr(UPOLOIPO);
%PILIKO=fliplr(PILIKO);
%---------------------------------------------------


[R,P,K]=residuez(num,den);
% keyboard
%OROS=PILIKO(1);
% Make rows columns
%R=R(:);P=P(:);K=K(:);
R=R(:);P=P(:);K=K(:);


% Distinction between Real & Image Residues  
[R,P,l_real,l_imag]=srtrp(R,P,'all');

% Construction of M(k) (Eq. 45 REF[1])
for k=1:length(P)
   ELEM=R./(ones(length(P),1)-P(k).*P);             % Construction of the terms Ri/1-pk*pi
   M(k)=R(k)*sum(ELEM);										 % Calculation of M(k)  
   clear ELEM
end

% Dispersion of Modes (Eq. 46 & 47 REF[1])
D_real=real(M(1:l_real));D_imag=M(l_real+1:l_imag+l_real);
D=[D_real';D_imag'+conj(D_imag)'];

% Precentage Dispersion (Eq. 48 REF[1])
%if ~isempty(K)
%   D=D(:).';
%   VARY=[K^2 2*K*OROS D]; 
%   Delta=100*VARY./sum(VARY);
	% tests   sum(Delta);Delta(1);Delta(2)
%   Delta=Delta(3:length(Delta))'
%else
%  disp('mhn mpeis')
	Delta=100*D./sum(D);
	%Delta=D_imag./sum(D_imag)
   sum(Delta);
   %dou=K^2/sum(D+K^2)
%end
%keyboard

% Sorting Dispersions by asceding Frequency 
lambda=P(l_real+1:l_imag+l_real);
Wn=Fs*abs(log(lambda));          % Corresponding Frequencies 
z= -cos(angle(log(lambda)));     % Damping Ratios
[Wn sr]=sort(Wn);
fn=Wn./(2*pi);                   % rad/sec==>Hz 
z=100*z(sr);

Delta=Delta(l_real+1:l_real+l_imag);
Delta=Delta(sr);

% Sorting Poles by asceding Frequency
lambda=lambda(sr);
R_imag_plus=R(l_real+1:l_real+l_imag);
R=R_imag_plus(sr);
%R=R.*Fs; 		% Residues for Impulse Invariance Method
%R=R./R(1);  	% Normalized Residues
   
Mag=abs(lambda);   % Magnitude of poles
Mag=Mag(sr);

%--------------------------------------------------------
% 				Results
%--------------------------------------------------------
form1= '%1d' ;
form2 = '%7.4e';  

if nargout==0,      
   % Print results on the screen. First generate corresponding strings:
   nmode = dprint([1:l_imag]','Mode',form1);
   wnstr = dprint(fn,'Frequency (Hz)',form2);
   zstr = dprint(z,'Damping (%)',form2);
   dstr = dprint(Delta,'Dispersion (%)',form2);
   rstr = dprint(R,'Norm. Residues ',form2);
   mrstr = dprint(lambda,'Poles',form1);
disp([nmode wnstr zstr dstr rstr mrstr	]);
else
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% From lti\damp.m
function s = dprint(v,str,form)
%DPRINT  Prints a column vector V with a centered caption STR on top

if isempty(v), 
   s = [];
   return
end

nv = length(v);
lrpad = char(' '*ones(nv+4,2));  % left and right spacing
lstr = length(str);

% Convert V to string
rev = real(v);
s = [blanks(nv)' num2str(abs(rev),form)];
s(rev<0,1) = '-';
if ~isreal(v),
   % Add complex part
   imv = imag(v);
   imags = num2str(abs(imv),[form 'i']);
   imags(~imv,:) = ' ';
   signs = char(' '*ones(nv,3));
   signs(imv>0,2) = '+';
   signs(imv<0,2) = '-';
   s = [s signs imags];
end

% Dimensions
ls = size(s,2);
lmax = max(ls,lstr);
ldiff = lstr - ls;
ldiff2 = floor(ldiff/2);
str = [blanks(-ldiff2) str blanks(-ldiff+ldiff2)];
s = [char(' '*ones(nv,ldiff2)) s char(' '*ones(nv,ldiff-ldiff2))];

% Put pieces together
s = [blanks(lmax) ; str ; blanks(lmax) ; s ; blanks(lmax)];
s = [lrpad s lrpad];

% end dprint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,P,nr,ni]=srtrp(R,P,flg)

% if flg='hf' ==> Real Residues & Imag Residues (from Real poles)
% else if flg='all' ==> Real Residues & Imag Residues (from positive poles) & Imag Residues (from negative Poles)

R_real=[];P_real=[];
R_imag_plus=[];P_imag_plus=[];
R_imag=[];P_imag=[];

for i=1:length(R)
   if imag(P(i))==0
      R_real=[R_real;R(i)];P_real=[P_real;P(i)];
   elseif imag(P(i))>0
      R_imag_plus=[R_imag_plus;R(i)];P_imag_plus=[P_imag_plus;P(i)];
   else
      R_imag=[R_imag;R(i)];P_imag=[P_imag;P(i)];
   end
end
switch flg
case 'all'
   R=[R_real;R_imag_plus;R_imag];P=[P_real;P_imag_plus;P_imag];
   nr=length(P_real);ni=length(P_imag);
case 'hf'
   P=[P_real;P_imag_plus];R=[R_real;R_imag_plus];
   nr=length(P_real);ni=length(P_imag);
end

function rk=acf_wn(x,maxlag,barsize)
% rk=acf_wn(x,maxlag,barsize);

R=xcorr(x,maxlag,'coeff');
rk=R(maxlag+2:end);
bar([1:maxlag],rk,barsize,'b'),hold
plot([1:maxlag],(1.96/sqrt(length(x))).*ones(maxlag,1),'r',[1:maxlag],(-1.96/sqrt(length(x))).*ones(maxlag,1),'r')
axis([0 maxlag+1 -1 1]),xlabel('Lag'),ylabel('A.C.F. ( \rho_\kappa )')
zoom on;hold

function varargout = autocorr(Series , nLags , Q , nSTDs)
%AUTOCORR Compute or plot sample auto-correlation function.
%   Compute or plot the sample auto-correlation function (ACF) of a univariate, 
%   stochastic time series. When called with no output arguments, AUTOCORR 
%   displays the ACF sequence with confidence bounds.
%
%   [ACF, Lags, Bounds] = autocorr(Series)
%   [ACF, Lags, Bounds] = autocorr(Series , nLags , M , nSTDs)
%
%   Optional Inputs: nLags , M , nSTDs
%
% Inputs:
%   Series - Vector of observations of a univariate time series for which the
%     sample ACF is computed or plotted. The last row of Series contains the
%     most recent observation of the stochastic sequence.
%
% Optional Inputs:
%   nLags - Positive, scalar integer indicating the number of lags of the ACF 
%     to compute. If empty or missing, the default is to compute the ACF at 
%     lags 0,1,2, ... T = minimum[20 , length(Series)-1]. Since an ACF is 
%     symmetric about zero lag, negative lags are ignored.
%
%   M - Non-negative integer scalar indicating the number of lags beyond which 
%     the theoretical ACF is deemed to have died out. Under the hypothesis that 
%     the underlying Series is really an MA(M) process, the large-lag standard
%     error is computed (via Bartlett's approximation) for lags > M as an 
%     indication of whether the ACF is effectively zero beyond lag M. On the 
%     assumption that the ACF is zero beyond lag M, Bartlett's approximation 
%     is used to compute the standard deviation of the ACF for lags > M. If M 
%     is empty or missing, the default is M = 0, in which case Series is 
%     assumed to be Gaussian white noise. If Series is a Gaussian white noise 
%     process of length N, the standard error will be approximately 1/sqrt(N).
%     M must be less than nLags.
%
%   nSTDs - Positive scalar indicating the number of standard deviations of the 
%     sample ACF estimation error to compute assuming the theoretical ACF of
%     Series is zero beyond lag M. When M = 0 and Series is a Gaussian white
%     noise process of length N, specifying nSTDs will result in confidence 
%     bounds at +/-(nSTDs/sqrt(N)). If empty or missing, default is nSTDs = 2 
%     (i.e., approximate 95% confidence interval).
%
% Outputs:
%   ACF - Sample auto-correlation function of Series. ACF is a vector of 
%     length nLags + 1 corresponding to lags 0,1,2,...,nLags. The first 
%     element of ACF is unity (i.e., ACF(1) = 1 = lag 0 correlation).
%
%   Lags - Vector of lags corresponding to ACF (0,1,2,...,nLags).
%
%   Bounds - Two element vector indicating the approximate upper and lower
%     confidence bounds assuming that Series is an MA(M) process. Note that 
%     Bounds is approximate for lags > M only.
%
% Example:
%   Create an MA(2) process from a sequence of 1000 Gaussian deviates, then 
%   visually assess whether the ACF is effectively zero for lags > 2:
%
%     randn('state',0)               % Start from a known state.
%     x = randn(1000,1);             % 1000 Gaussian deviates ~ N(0,1).
%     y = filter([1 -1 1] , 1 , x);  % Create an MA(2) process.
%     autocorr(y , [] , 2)           % Inspect the ACF with 95% confidence.
%
% See also CROSSCORR, PARCORR, FILTER.

%   Copyright 1999-2003 The MathWorks, Inc.   
%   $Revision: 1.6.2.1 $  $Date: 2003/05/08 21:45:15 $

%
% Reference:
%   Box, G.E.P., Jenkins, G.M., Reinsel, G.C., "Time Series Analysis: 
%     Forecasting and Control", 3rd edition, Prentice Hall, 1994.

%
% Ensure the sample data is a VECTOR.
%

[rows , columns]  =  size(Series);

if (rows ~= 1) & (columns ~= 1) 
    error('GARCH:autocorr:NonVectorInput' , ' Input ''Series'' must be a vector.');
end

rowSeries   =  size(Series,1) == 1;

Series      =  Series(:);       % Ensure a column vector
n           =  length(Series);  % Sample size.
defaultLags =  20;              % BJR recommend about 20 lags for ACFs.

%
% Ensure the number of lags, nLags, is a positive 
% integer scalar and set default if necessary.
%

if (nargin >= 2) & ~isempty(nLags)
   if prod(size(nLags)) > 1
      error('GARCH:autocorr:NonScalarLags' , ' Number of lags ''nLags'' must be a scalar.');
   end
   if (round(nLags) ~= nLags) | (nLags <= 0)
      error('GARCH:autocorr:NonPositiveInteger' , ' Number of lags ''nLags'' must be a positive integer.');
   end
   if nLags > (n - 1)
      error('GARCH:autocorr:LagsTooLarge' , ' Number of lags ''nLags'' must not exceed ''Series'' length - 1.');
   end
else
   nLags  =  min(defaultLags , n - 1);
end

%
% Ensure the hypothesized number of lags, Q, is a non-negative integer
% scalar, and set default if necessary.
%
if (nargin >= 3) & ~isempty(Q)
   if prod(size(Q)) > 1
      error('GARCH:autocorr:NonScalarQ' , ' Number of lags ''Q'' must be a scalar.');
   end
   if (round(Q) ~= Q) | (Q < 0)
      error('GARCH:autocorr:NegativeInteger' , ' Number of lags ''Q'' must be a non-negative integer.');
   end
   if Q >= nLags
      error('GARCH:autocorr:QTooLarge' , ' ''Q'' must be less than ''nLags''.');
   end
else
   Q  =  0;     % Default is 0 (Gaussian white noise hypothisis).
end

%
% Ensure the number of standard deviations, nSTDs, is a positive 
% scalar and set default if necessary.
%

if (nargin >= 4) & ~isempty(nSTDs)
   if prod(size(nSTDs)) > 1
      error('GARCH:autocorr:NonScalarSTDs' , ' Number of standard deviations ''nSTDs'' must be a scalar.');
   end
   if nSTDs < 0
      error('GARCH:autocorr:NegativeSTDs' , ' Number of standard deviations ''nSTDs'' must be non-negative.');
   end
else
   nSTDs =  2;     % Default is 2 standard errors (95% condfidence interval).
end

%
% Convolution, polynomial multiplication, and FIR digital filtering are
% all the same operation. The FILTER command could be used to compute 
% the ACF (by computing the correlation by convolving the de-meaned 
% Series with a flipped version of itself), but FFT-based computation 
% is significantly faster for large data sets.
%
% The ACF computation is based on Box, Jenkins, Reinsel, pages 30-34, 188.
%

nFFT =  2^(nextpow2(length(Series)) + 1);
F    =  fft(Series-mean(Series) , nFFT);
F    =  F .* conj(F);
ACF  =  ifft(F);
ACF  =  ACF(1:(nLags + 1));         % Retain non-negative lags.
ACF  =  ACF ./ ACF(1);     % Normalize.
ACF  =  real(ACF);

%
% Compute approximate confidence bounds using the Box-Jenkins-Reinsel 
% approach, equations 2.1.13 and 6.2.2, on pages 33 and 188, respectively.
%

sigmaQ  =  sqrt((1 + 2*(ACF(2:Q+1)'*ACF(2:Q+1)))/n);  
bounds  =  sigmaQ * [nSTDs ; -nSTDs];
Lags    =  [0:nLags]';

if nargout == 0                     % Make plot if requested.

%
%  Plot the sample ACF.
%
   lineHandles  =  stem(Lags , ACF , 'filled' , 'r-o');
   set   (lineHandles(1) , 'MarkerSize' , 4)
   grid  ('on')
   xlabel('Lag')
   ylabel('Sample Autocorrelation')
   title ('Sample Autocorrelation Function (ACF)')
   hold  ('on')
%
%  Plot the confidence bounds under the hypothesis that the underlying 
%  Series is really an MA(Q) process. Bartlett's approximation gives
%  an indication of whether the ACF is effectively zero beyond lag Q. 
%  For this reason, the confidence bounds (horizontal lines) appear 
%  over the ACF ONLY for lags GREATER than Q (i.e., Q+1, Q+2, ... nLags).
%  In other words, the confidence bounds enclose ONLY those lags for 
%  which the null hypothesis is assumed to hold. 
%

   plot([Q+0.5 Q+0.5 ; nLags nLags] , [bounds([1 1]) bounds([2 2])] , '-b');

   plot([0 nLags] , [0 0] , '-k');
   hold('off')
   a  =  axis;
   axis([a(1:3) 1]);

else

%
%  Re-format outputs for compatibility with the SERIES input. When SERIES is
%  input as a row vector, then pass the outputs as a row vectors; when SERIES
%  is a column vector, then pass the outputs as a column vectors.
%
   if rowSeries
      ACF     =  ACF.';
      Lags    =  Lags.';
      bounds  =  bounds.';
   end

   varargout  =  {ACF , Lags , bounds};

end

function [modal]=the_modals(num,den,Fs,s,m)
% Estimation of the modal parameters (works inside the monte_carlo.m)
for i=1:s*m;
   [Delta,fn,z] = disper_new120106(num(i,:),den,Fs);
   Delta_full(:,i) = Delta;
   clear Delta
end

Delta_na=[];
for i = 1:length(fn)
   Delta_na(i,:) = max(abs(Delta_full(i,:)));
end
modal = [fn z Delta_na];

function [Delta,fn,z,R,lambda]=disper_new120106(num,den,Fs)

% num		: The numerator of the transfer function
% den		: The denominator of the transfer function
% Fs		: The sampling frequency (Hz)
% Delta	: The precentage dispersion
% fn		: The corresponding frequencies (Hz)
% z		: The corresponding damping (%)
% R		: The residues of the discrete system
% Mag		: The magnitude of the corresponding poles
% This function computes the dispersion of each frequency of a system. The System is  
% enetred as a transfer function. In case the order of numerator polynomial is greater than 
% that of the denominator the polynomial division is apllied, and the dispersion is considered at
% the remaine tf. The analysis is done using the Residuez routine of MATLAB.
% The results are printed in the screen in asceding order of frequencies.
% This routine displays only the dispersions from the natural frequencies (Complex Poles).

% REFERENCE[1]:  MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES - A Critical Assessment 
% REFERENCE[2]:  PANDIT WU

%--------------------------------------------
% Created	: 08 December 1999.
% Author(s)	: A. Florakis & K.A.Petsounis
% Updated	: 17 February 2006.
%--------------------------------------------

% Sampling Period
Ts=1/Fs;

% Calculate the residues of the Transfer Function
num=num(:).';
den=den(:).';

[R,P,K]=residuez(num,den);

R=R(:);P=P(:);K=K(:);


% Distinction between Real & Image Residues  
[R,P,l_real,l_imag]=srtrp(R,P,'all');

% Construction of M(k) (Eq. 45 REF[1])
for k=1:length(P)
   ELEM=R./(ones(length(P),1)-P(k).*P);             % Construction of the terms Ri/1-pk*pi
   M(k)=R(k)*sum(ELEM);										 % Calculation of M(k)  
   clear ELEM
end

% Dispersion of Modes (Eq. 46 & 47 REF[1])
D_real=real(M(1:l_real));D_imag=M(l_real+1:l_imag+l_real);
D=[D_real';D_imag'+conj(D_imag)'];


Delta=100*D./sum(D);     % Delta (%) 

% Sorting Dispersions by asceding Frequency 
lambda=P(l_real+1:l_imag+l_real);
Wn=Fs*abs(log(lambda));          % Corresponding Frequencies 
z= -cos(angle(log(lambda)));     % Damping Ratios
[Wn sr]=sort(Wn);
fn=Wn./(2*pi);                   % fn rad/sec==>Hz 
z=100*z(sr);                     % damping ratio(%) 

Delta=Delta(l_real+1:l_real+l_imag);
Delta=Delta(sr);

% Sorting Poles by asceding Frequency
lambda=lambda(sr);
R_imag_plus=R(l_real+1:l_real+l_imag);
R=R_imag_plus(sr);
%R=R.*Fs; 		% Residues for Impulse Invariance Method
%R=R./R(1);  	% Normalized Residues
   
Mag=abs(lambda);   % Magnitude of poles
Mag=Mag(sr);

%--------------------------------------------------------
% 				Results
%--------------------------------------------------------
form1= '%1d' ;
form2 = '%7.4e';  

if nargout==0   
   % Print results on the screen. First generate corresponding strings:
   nmode = dprint([1:l_imag]','Mode',form1);
   wnstr = dprint(fn,'Frequency (Hz)',form2);
   zstr = dprint(z,'Damping (%)',form2);
   dstr = dprint(Delta,'Dispersion (%)',form2);
   rstr = dprint(R,'Norm. Residues ',form2);
   mrstr = dprint(lambda,'Poles',form1);
disp([nmode wnstr zstr dstr rstr mrstr	]);
else
end

%--------------------------------------------------------------------------
% %------------------------------------------------------------------------
% %                                THE END
% %------------------------------------------------------------------------
% %------------------------------------------------------------------------
