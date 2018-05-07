%% ECE 160 - CHALLENGE 2: "Discriminating Speech From Music" %%

%NAME 1: ANTONIO JAVIER SAMANIEGO JURADO, PERM: 6473680
%NAME 2: PABLO MARTIN GARCIA, PERM: 6473607

clear all
close all


%% Load Features: %%
%Loads variables when the algorithm is already trained and only aims to
%classify input audio files 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% UNCOMMENT IF ALGORITHM TRAINING WANTS TO BE PERFORMED): %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load('evaluation_var.mat')
%load('plot_var.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Training data: %%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  COMMENT TRAINING DATA SECTION BELOW IF ONLY AUDIO CLASSIFICATION IS DESIRED:  %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



disp('Training data...');

%%%%% Set number of files to be extracted features from below: %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_files=39;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs_audio=zeros(num_files,1);
fs_speech=zeros(num_files,1);

%Music:
disp('Reading music...');

C_music=cell(1,40);
C_music{40}=zeros(num_files,1); %Position 40 of the cell corresponds to all sampling freq values.
 
path='./audio/music'; 
folder = path; 
dirListing = dir(folder);


%Trim directory for possible garbage files:
if strcmp(dirListing(1).name, '.') 
   dirListing(1) = [];  
end
if strcmp(dirListing(1).name, '..') 
   dirListing(1) = [];
end
if strcmp(dirListing(1).name, '.DS_Store') 
   dirListing(1) = [];
end


for i=1:length(dirListing)
    fileName = fullfile(folder,dirListing(i).name); 
    [audio,fs_audio]=audioread(fileName);
    C_music{i}=audio(:,1);
    C_music{40}(i)=fs_audio;
end

disp('Music read and loaded');

%Speech:
disp('Reading speech...');

C_speech=cell(1,40);
C_speech{40}=zeros(num_files,1);

path='./audio/speech';
folder = path; 
dirListing = dir(folder);

%Trim directory for possible garbage files:
if strcmp(dirListing(1).name, '.') 
   dirListing(1) = [];  
end
if strcmp(dirListing(1).name, '..') 
   dirListing(1) = [];
end
if strcmp(dirListing(1).name, '.DS_Store') 
   dirListing(1) = [];
end


for i=1:length(dirListing)
    fileName = fullfile(folder,dirListing(i).name); 
    [audio,fs_audio]=audioread(fileName); 
    C_speech{i}=audio(:,1);
    C_speech{40}(i)=fs_audio;
end

disp('Speech read and loaded');

%}

%% Extract Features from training data: %%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  COMMENT FEATURE EXTRACTION SECTION BELOW IF ONLY AUDIO CLASSIFICATION IS DESIRED:  %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TIME DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% SILENCE:
        disp('Calculating silence in data...');
        % Music
        prop_silence_music = zeros(39,1);
        number = 5;
        maxim = zeros(number,1);
        for i=1:39
            len = length(C_music{1,i});
            for j=1:length(maxim)
                maxim(j) = max(C_music{1,i}((j-1)*round(len/number)+1:j*round(len/number)));
            end
            maximus = mean(maxim);
            prop_silence_music(i) = sum(C_music{1,i}<(0.1*maximus))/length(C_music{1,i});
        end
        prop_silence_music_ave = mean(prop_silence_music);
        % Speech
        prop_silence_speech = zeros(39,1);
        for i=1:39
            len = length(C_speech{1,i});
            for j=1:length(maxim)
                maxim(j) = max(C_speech{1,i}((j-1)*round(len/number)+1:j*round(len/number)));
            end
            maximus = mean(maxim);
            prop_silence_speech(i) = sum(C_speech{1,i}<(0.1*maximus))/length(C_speech{1,i});
        end
        prop_silence_speech_ave = mean(prop_silence_speech);

        vectorsilencemusic = zeros(size(prop_silence_music));
        vectorsilencemusic(:) = prop_silence_music_ave;
        vectorsilencespeech = zeros(size(prop_silence_speech));
        vectorsilencespeech(:) = prop_silence_speech_ave;
        disp('Silence in data calculated');

               
        
        
%VARIANCE:
        disp('Calculating variance of training data...');
 
        variances_music=zeros(num_files,1);
        variances_speech=zeros(num_files,1);

        for i=1:num_files
            variances_music(i)=var(C_music{i}(:));
            variances_speech(i)=var(C_speech{i}(:));
        end

        variances_music_av=mean(variances_music);
        variances_speech_av=mean(variances_speech);

        disp('Mean and Variance of training data calcualted');





%ZERO-CROSSING:
        disp('Calculating zero-crossing of training data...');
        zero_crossing_music=zeros(num_files,1);
        zero_crossing_speech=zeros(num_files,1);

        for i=1:num_files
            zero_crossing_music(i,1)=0;
            for j=2:length(C_music{i}(:))
                if C_music{i}(j)*C_music{i}(j-1)<0
                    zero_crossing_music(i,1)=zero_crossing_music(i,1)+1;
                end
            end
            zero_crossing_speech(i,1)=0;
            for j=2:length(C_speech{i}(:))
                if C_speech{i}(j)*C_speech{i}(j-1)<0
                    zero_crossing_speech(i,1)=zero_crossing_speech(i,1)+1;
                end
            end 
        end
        zero_crossing_music_av=mean(zero_crossing_music);
        zero_crossing_speech_av=mean(zero_crossing_speech);
        disp('Zero-crossing of training data calculated');




%AUTOCORRELATION:
        disp('Calculating autocorrelation of training data...');
        % Music
        autocorr_music=cell(2,num_files);
        for i=1:length(C_music)-1
            autocorr_music{1,i}=xcorr(C_music{1,i});
            autocorr_music{2,i}=mean(autocorr_music{1,i});
        end

        media_music=0;
        for i=1:length(autocorr_music)
            media_music=media_music+autocorr_music{2,i};
        end
        media_music_autocorr=media_music/length(autocorr_music);

        % Speech
        autocorr_speech = cell(2,num_files);
        for i=1:length(C_speech)-1
            autocorr_speech{1,i}=xcorr(C_speech{1,i});
            autocorr_speech{2,i}=mean(autocorr_speech{1,i});
        end

        media_speech=0;
        for i=1:length(autocorr_speech)
            media_speech=media_speech+autocorr_speech{2,i};
        end
        media_speech_autocorr=media_speech/length(autocorr_speech);
        
        disp('Autocorrelation of training data calculated');

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% FREQUENCY DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%FFT:
        %Music:
        S_music=cell(3,num_files); %S{1} contains fft values, S{2} contains spectrum freq values, and S{3} contains abs(fft) values.

        for i=1:num_files
            NFFT1=2^nextpow2(length(C_music{i}(:)));
            S_music{1,i}(:)=fft(C_music{i}(:),NFFT1)/length(C_music{i}(:));
            S_music{2,i}(:)=C_music{40}(i)/2*linspace(0,1,NFFT1/2+1);
            S_music{3,i}(:)=2*abs(S_music{1,i}(1:NFFT1/2+1));
        end
        
        %Speech:
        S_speech=cell(3,num_files); %S{1} contains fft values, S{2} contains spectrum freq values, and S{3} contains abs(fft) values.

        for i=1:num_files
            NFFT1=2^nextpow2(length(C_speech{i}(:)));
            S_speech{1,i}(:)=fft(C_speech{i}(:),NFFT1)/length(C_speech{i}(:));
            S_speech{2,i}(:)=C_speech{40}(i)/2*linspace(0,1,NFFT1/2+1);
            S_speech{3,i}(:)=2*abs(S_speech{1,i}(1:NFFT1/2+1));
        end


        
        
%RMS (over 6 different frequency spectrum intervals):
        
        disp('Calculating RMS of training data...');

        RMS_global=cell(2,num_files); %RMS_global{1} contains music and RMS_global{2} contains speech.
        num_intervals=6;
        
        RMS_music_ave=zeros(6,1);
        RMS_speech_ave=zeros(6,1);
        
        for k=1:num_files
            
            spectrum_index_music=1:length(S_music{2,k}(:));
            spectrum_index_speech=1:length(S_speech{2,k}(:));
            
            RMS_music=zeros(num_intervals,1);
            RMS_speech=zeros(num_intervals,1);
            
            interval1_music=spectrum_index_music(1):round(0.1*spectrum_index_music(end));
            interval2_music=round(0.1*spectrum_index_music(end)):round(0.2*spectrum_index_music(end));
            interval3_music=round(0.2*spectrum_index_music(end)):round(0.3*spectrum_index_music(end));
            interval4_music=round(0.3*spectrum_index_music(end)):round(0.4*spectrum_index_music(end));
            interval5_music=round(0.4*spectrum_index_music(end)):round(0.7*spectrum_index_music(end));
            interval6_music=round(0.7*spectrum_index_music(end)):spectrum_index_music(end);

            interval1_speech=spectrum_index_speech(1):round(0.1*spectrum_index_speech(end));
            interval2_speech=round(0.1*spectrum_index_speech(end)):round(0.2*spectrum_index_speech(end));
            interval3_speech=round(0.2*spectrum_index_speech(end)):round(0.3*spectrum_index_speech(end));
            interval4_speech=round(0.3*spectrum_index_speech(end)):round(0.4*spectrum_index_speech(end));
            interval5_speech=round(0.4*spectrum_index_speech(end)):round(0.7*spectrum_index_speech(end));
            interval6_speech=round(0.7*spectrum_index_speech(end)):spectrum_index_speech(end);          
           
            %First interval RMS:
            aux1=sum(S_music{3,k}(interval1_music).^2);
            RMS_music(1)=sqrt(aux1/length(aux1));
            aux1=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(1)=sqrt(aux1/length(aux1));
            
            %Second interval RMS:
            aux2=sum(S_music{3,k}(interval2_music).^2);
            RMS_music(2)=sqrt(aux2/length(aux2));
            aux2=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(2)=sqrt(aux2/length(aux2));
            
            %Third interval RMS:
            aux3=sum(S_music{3,k}(interval3_music).^2);
            RMS_music(3)=sqrt(aux3/length(aux3));
            aux4=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(3)=sqrt(aux3/length(aux3));
            
            %Fourth interval RMS:
            aux4=sum(S_music{3,k}(interval4_music).^2);
            RMS_music(4)=sqrt(aux4/length(aux4));
            aux4=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(4)=sqrt(aux4/length(aux4));
            
            %Fifth interval RMS:
            aux5=sum(S_music{3,k}(interval5_music).^2);
            RMS_music(5)=sqrt(aux5/length(aux5));
            aux5=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(5)=sqrt(aux5/length(aux5));
            
            %Sixth interval RMS:
            aux6=sum(S_music{3,k}(interval6_music).^2);
            RMS_music(6)=sqrt(aux6/length(aux6));
            aux6=sum(S_speech{3,k}(interval1_speech).^2);
            RMS_speech(6)=sqrt(aux6/length(aux6));
           
            RMS_global{1,k}(:)=RMS_music;
            RMS_global{2,k}(:)=RMS_speech;  

        end
        
        for n=1:num_intervals
            sum_music=0;
            sum_speech=0;
            
            for m=1:39
                sum_music=sum_music+RMS_global{1,m}(n);
                sum_speech=sum_speech+RMS_global{2,m}(n);
            end 
            
            RMS_music_ave(n)=sum_music/num_intervals;
            RMS_speech_ave(n)=sum_speech/num_intervals;
        end


        disp('RMS of training data calculated');

        
        
        
%SPECTRAL CENTROID:

        disp('Calculating Spectral Centroid of training data...');
        %Music:
        spectral_centroid_music=zeros(num_files,1);
        
        for i=1:num_files

            S1_music=S_music{1,i}(:);
            
            SIGNAL_music = abs(S1_music);
            normalized_spectrum_music = SIGNAL_music/sum(SIGNAL_music);
            normalized_frequencies_music = linspace(0,1,length(SIGNAL_music));
            spectral_centroid_music(i) = sum(normalized_frequencies_music * normalized_spectrum_music);
        end
        spectral_centroid_music_ave=mean(spectral_centroid_music);

        
        %Speech:
        spectral_centroid_speech=zeros(num_files,1);
        for i=1:num_files

            S1_speech=S_speech{1,i}(:);

            SIGNAL_speech = abs(S1_speech);
            normalized_spectrum_speech = SIGNAL_speech/sum(SIGNAL_speech);
            normalized_frequencies_speech = linspace(0,1,length(SIGNAL_speech));
            spectral_centroid_speech(i) = sum(normalized_frequencies_speech * normalized_spectrum_speech);
        end
        spectral_centroid_speech_ave=mean(spectral_centroid_speech);

        disp('Spectral Centroid of training data calculated');



%% Read, process, and classify input audio files: %%
  
fprintf("\n")
disp('Starting processing/classification for input audio file(s)...');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% SPECIFY PATH WHERE INPUT AUDIO FILES TO BE CLASSIFIED ARE LOCATED: %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path='./test_files'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder = path; 
dirListing = dir(folder);
  

%Trim directory for possible garbage files:
if strcmp(dirListing(1).name, '.') 
   dirListing(1) = [];  
end
if strcmp(dirListing(1).name, '..') 
   dirListing(1) = [];
end
if strcmp(dirListing(1).name, '.DS_Store') 
   dirListing(1) = [];
end

  
music_speech=cell(length(dirListing),2);


for m=1:length(dirListing)
    
        fileName = fullfile(folder,dirListing(m).name); 
        [input_audio,fs_audio]=audioread(fileName);

        music_speech{m,1}=dirListing(m).name;
        
        Fs=fs_audio;

        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% TIME DOMAIN: %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%

        % Silence:
        number = 5;
        maxim = zeros(number,1);
        len = length(input_audio);
        for j=1:length(maxim)
            maxim(j) = max(input_audio((j-1)*round(len/number)+1:j*round(len/number)));
        end
        maximus = mean(maxim);
        prop_silence = sum(input_audio<(0.1*maximus))/length(input_audio);

        %Mean & Variance:
        variance=var(input_audio);

        %Zero-crossing:
        zero_crossing_input=0;
        for i=2:length(input_audio)
            if input_audio(i)*input_audio(i-1)<0
                zero_crossing_input=zero_crossing_input+1;
            end
        end   

        %Autocorrelation:
        autocorr=xcorr(input_audio);
        autocorr_mean=mean(autocorr);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% FREQUENCY DOMAIN: %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        NFFT1=2^nextpow2(length(input_audio));
        S1=fft(input_audio,NFFT1)/length(input_audio);
        freq=Fs/2*linspace(0,1,NFFT1/2+1);
        freq_values=2*abs(S1(1:NFFT1/2+1));

        %RMS:
        spectrum_index=1:length(freq);
        num_intervals=6;
        RMS=zeros(num_intervals,1);

        interval1=spectrum_index(1):round(0.1*spectrum_index(end));
        interval2=round(0.1*spectrum_index(end)):round(0.2*spectrum_index(end));
        interval3=round(0.2*spectrum_index(end)):round(0.3*spectrum_index(end));
        interval4=round(0.3*spectrum_index(end)):round(0.4*spectrum_index(end));
        interval5=round(0.4*spectrum_index(end)):round(0.7*spectrum_index(end));
        interval6=round(0.7*spectrum_index(end)):spectrum_index(end);

        %First interval RMS:
        aux1=sum(freq_values(interval1).^2);
        RMS(1)=sqrt(aux1/length(aux1));

        %Second interval RMS:
        aux2=sum(freq_values(interval2).^2);
        RMS(2)=sqrt(aux2/length(aux2));

        %Third interval RMS:
        aux3=sum(freq_values(interval3).^2);
        RMS(3)=sqrt(aux3/length(aux3));

        %Fourth interval RMS:
        aux4=sum(freq_values(interval4).^2);
        RMS(4)=sqrt(aux4/length(aux4));

        %Fifth interval RMS:
        aux5=sum(freq_values(interval5).^2);
        RMS(5)=sqrt(aux5/length(aux5));

        %Sixth interval RMS:
        aux6=sum(freq_values(interval6).^2);
        RMS(6)=sqrt(aux6/length(aux6));

        RMS=mean(RMS);



        %SPECTRAL CENTROID:
        SIGNAL = abs(S1);
        normalized_spectrum = SIGNAL/sum(SIGNAL);
        normalized_frequencies = linspace(0,1,length(SIGNAL));
        spectral_centroid = sum(normalized_frequencies * normalized_spectrum);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Classification, based on parameters: %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        music=0;
        speech=0;

        %Silence Classification:
        if prop_silence<0.6
            music=music+1.5;
        elseif (0.6<prop_silence) && (prop_silence<0.65)
            music=music+1;
        elseif (0.65<prop_silence) && (prop_silence<0.7)
            music=music+0.25;
        elseif (0.8<prop_silence) && (prop_silence<0.85)
            speech=speech+1;
        elseif prop_silence>0.85
            speech=speech+2; 
        end

        %RMS Classification:  
        if (RMS<0.2) || (RMS>0.8)
            music=music+1.5;
        elseif (RMS>0.3) && (RMS<0.4)
            music=music+1.2;
        elseif (RMS>0.4) && (RMS<0.8)
            speech=speech+2;
        elseif (RMS>0.2) && (RMS<0.3)
            speech=speech+2.5;
            music=music+0.3;
        end    


        %Variance Classification:
        if variance>variances_music_av
            music=music+1;
        elseif variance<variances_speech_av
            speech=speech+1.5;
        end

        %Zero-Crossing Classification:
        if zero_crossing_input>zero_crossing_speech_av
            speech=speech+2;
        elseif zero_crossing_input<zero_crossing_music_av
            music=music+1;
        end

        %Autocorrelation Classification:
        if abs(autocorr_mean-media_music_autocorr)<abs(autocorr_mean-media_speech_autocorr)
            music=music+0.1;
        else
            speech=speech+0.1;
        end

        %Spectral Centroid Classification:
        if spectral_centroid>spectral_centroid_speech_ave
            speech=speech+1.5;
        elseif spectral_centroid<spectral_centroid_music_ave
            music=music+1.5;
        end


        %RESULT: 0 means failed, 1 means success, 0.5 means blurred line
        %between both:
        if music<speech
            result=1;
        elseif music>speech
            result=0;
        else
            result=0.5;   
        end

        music_speech{m,2}=result;
        
end

fprintf("\n")
disp('Finished');
fprintf('\n(Note: In case some output was 0.5, it means it was not clear whether the audio was music or speech).\n\n')

music_speech
        
%% Evaluate accuracy of the algorithm, based on available data (files): %%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  COMMENT ACCURACY EVALUATION SECTION BELOW IF ONLY AUDIO CLASSIFICATION IS DESIRED:  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('Evaluating algorithm based on features from # of files available...');

result_music=zeros(num_files,1);

for k=1:num_files
    
    input_audio=C_music{k}(:);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TIME DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Silence:
    number = 5;
    maxim = zeros(number,1);
    len = length(input_audio);
    for j=1:length(maxim)
        maxim(j) = max(input_audio((j-1)*round(len/number)+1:j*round(len/number)));
    end
    maximus = mean(maxim);
    prop_silence = sum(input_audio<(0.1*maximus))/length(input_audio);
    
    %Mean & Variance:
    variance=var(input_audio);

    %Zero-crossing:
    zero_crossing_input=0;
    for i=2:length(input_audio)
        if input_audio(i)*input_audio(i-1)<0
            zero_crossing_input=zero_crossing_input+1;
        end
    end   

    %Autocorrelation:
    autocorr=xcorr(input_audio);
    autocorr_mean=mean(autocorr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% FREQUENCY DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    NFFT1=2^nextpow2(length(input_audio));
    S1=fft(input_audio,NFFT1)/length(input_audio);
    freq=C_music{40}(k)/2*linspace(0,1,NFFT1/2+1);
    freq_values=2*abs(S1(1:NFFT1/2+1));

    %RMS:
            spectrum_index=1:length(freq);
            num_intervals=6;
            RMS=zeros(num_intervals,1);

            interval1=spectrum_index(1):round(0.1*spectrum_index(end));
            interval2=round(0.1*spectrum_index(end)):round(0.2*spectrum_index(end));
            interval3=round(0.2*spectrum_index(end)):round(0.3*spectrum_index(end));
            interval4=round(0.3*spectrum_index(end)):round(0.4*spectrum_index(end));
            interval5=round(0.4*spectrum_index(end)):round(0.7*spectrum_index(end));
            interval6=round(0.7*spectrum_index(end)):spectrum_index(end);

            %First interval RMS:
            aux1=sum(freq_values(interval1).^2);
            RMS(1)=sqrt(aux1/length(aux1));

            %Second interval RMS:
            aux2=sum(freq_values(interval2).^2);
            RMS(2)=sqrt(aux2/length(aux2));

            %Third interval RMS:
            aux3=sum(freq_values(interval3).^2);
            RMS(3)=sqrt(aux3/length(aux3));

            %Fourth interval RMS:
            aux4=sum(freq_values(interval4).^2);
            RMS(4)=sqrt(aux4/length(aux4));

            %Fifth interval RMS:
            aux5=sum(freq_values(interval5).^2);
            RMS(5)=sqrt(aux5/length(aux5));

            %Sixth interval RMS:
            aux6=sum(freq_values(interval6).^2);
            RMS(6)=sqrt(aux6/length(aux6));
            
            RMS=mean(RMS);



    %SPECTRAL CENTROID:
    SIGNAL = abs(S1);
    normalized_spectrum = SIGNAL/sum(SIGNAL);
    normalized_frequencies = linspace(0,1,length(SIGNAL));
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Classification, based on parameters: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    music=0;
    speech=0;
    
    %Silence Classification:
    if prop_silence<0.6
        music=music+1.5;
    elseif (0.6<prop_silence) && (prop_silence<0.65)
        music=music+1;
    elseif (0.65<prop_silence) && (prop_silence<0.7)
        music=music+0.25;
    elseif (0.8<prop_silence) && (prop_silence<0.85)
        speech=speech+1;
    elseif prop_silence>0.85
        speech=speech+2; 
    end
    
    %RMS Classification:  
    if (RMS<0.2) || (RMS>0.8)
        music=music+1.5;
    elseif (RMS>0.3) && (RMS<0.4)
        music=music+1.2;
    elseif (RMS>0.4) && (RMS<0.8)
        speech=speech+2;
    elseif (RMS>0.2) && (RMS<0.3)
        speech=speech+2.5;
        music=music+0.3;
    end    
         
    
    %Variance Classification:
    if variance>variances_music_av
        music=music+1;
    elseif variance<variances_speech_av
        speech=speech+1.5;
    end

    %Zero-Crossing Classification:
    if zero_crossing_input>zero_crossing_speech_av
        speech=speech+2;
    elseif zero_crossing_input<zero_crossing_music_av
        music=music+1;
    end
    
    %Autocorrelation Classification:
    if abs(autocorr_mean-media_music_autocorr)<abs(autocorr_mean-media_speech_autocorr)
        music=music+0.1;
    else
        speech=speech+0.1;
    end

    %Spectral Centroid Classification:
    if spectral_centroid>spectral_centroid_speech_ave
        speech=speech+1.5;
    elseif spectral_centroid<spectral_centroid_music_ave
        music=music+1.5;
    end

    
    
    %RESULT: 0 means failed, 0.5 means blurred line, and 1 means success.
    if music<speech
        result_music(k)=0;
    elseif music>speech
        result_music(k)=1;
    else 
        result_music(k)=0.5;
    end
    
end

result_speech=zeros(num_files,1);

for k=1:num_files
    
    input_audio=C_speech{k}(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TIME DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Silence:
    number = 5;
    maxim = zeros(number,1);
    len = length(input_audio);
    for j=1:length(maxim)
        maxim(j) = max(input_audio((j-1)*round(len/number)+1:j*round(len/number)));
    end
    maximus = mean(maxim);
    prop_silence = sum(input_audio<(0.1*maximus))/length(input_audio);
    
    %Mean & Variance:
    variance=var(input_audio);

    %Zero-crossing:
    zero_crossing_input=0;
    for i=2:length(input_audio)
        if input_audio(i)*input_audio(i-1)<0
            zero_crossing_input=zero_crossing_input+1;
        end
    end   

    %Autocorrelation:
    autocorr=xcorr(input_audio);
    autocorr_mean=mean(autocorr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% FREQUENCY DOMAIN: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    NFFT1=2^nextpow2(length(input_audio));
    S1=fft(input_audio,NFFT1)/length(input_audio);
    freq=C_music{40}(k)/2*linspace(0,1,NFFT1/2+1);
    freq_values=2*abs(S1(1:NFFT1/2+1));
    
    
    %RMS:
            spectrum_index=1:length(freq);
            num_intervals=6;
            RMS=zeros(num_intervals,1);

            interval1=spectrum_index(1):round(0.1*spectrum_index(end));
            interval2=round(0.1*spectrum_index(end)):round(0.2*spectrum_index(end));
            interval3=round(0.2*spectrum_index(end)):round(0.3*spectrum_index(end));
            interval4=round(0.3*spectrum_index(end)):round(0.4*spectrum_index(end));
            interval5=round(0.4*spectrum_index(end)):round(0.7*spectrum_index(end));
            interval6=round(0.7*spectrum_index(end)):spectrum_index(end);

            %First interval RMS:
            aux1=sum(freq_values(interval1).^2);
            RMS(1)=sqrt(aux1/length(aux1));

            %Second interval RMS:
            aux2=sum(freq_values(interval2).^2);
            RMS(2)=sqrt(aux2/length(aux2));

            %Third interval RMS:
            aux3=sum(freq_values(interval3).^2);
            RMS(3)=sqrt(aux3/length(aux3));

            %Fourth interval RMS:
            aux4=sum(freq_values(interval4).^2);
            RMS(4)=sqrt(aux4/length(aux4));

            %Fifth interval RMS:
            aux5=sum(freq_values(interval5).^2);
            RMS(5)=sqrt(aux5/length(aux5));

            %Sixth interval RMS:
            aux6=sum(freq_values(interval6).^2);
            RMS(6)=sqrt(aux6/length(aux6));
            
            RMS=mean(RMS);



    %SPECTRAL CENTROID:
    SIGNAL = abs(S1);
    normalized_spectrum = SIGNAL/sum(SIGNAL);
    normalized_frequencies = linspace(0,1,length(SIGNAL));
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Classification, based on parameters: %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    music=0;
    speech=0;
    
    %Silence Classification:
    if prop_silence<0.6
        music=music+1.5;
    elseif (0.6<prop_silence) && (prop_silence<0.65)
        music=music+1;
    elseif (0.65<prop_silence) && (prop_silence<0.7)
        music=music+0.25;
    elseif (0.8<prop_silence) && (prop_silence<0.85)
        speech=speech+1;
    elseif prop_silence>0.85
        speech=speech+2; 
    end
    
    %RMS Classification:  
    if (RMS<0.2) || (RMS>0.8)
        music=music+1.5;
    elseif (RMS>0.3) && (RMS<0.4)
        music=music+1.2;
    elseif (RMS>0.4) && (RMS<0.8)
        speech=speech+2;
    elseif (RMS>0.2) && (RMS<0.3)
        speech=speech+2.5;
        music=music+0.3;
    end    
         
    
    %Variance Classification:
    if variance>variances_music_av
        music=music+1;
    elseif variance<variances_speech_av
        speech=speech+1.5;
    end

    %Zero-Crossing Classification:
    if zero_crossing_input>zero_crossing_speech_av
        speech=speech+2;
    elseif zero_crossing_input<zero_crossing_music_av
        music=music+1;
    end
    
    %Autocorrelation Classification:
    if abs(autocorr_mean-media_music_autocorr)<abs(autocorr_mean-media_speech_autocorr)
        music=music+0.1;
    else
        speech=speech+0.1;
    end

    %Spectral Centroid Classification:
    if spectral_centroid>spectral_centroid_speech_ave
        speech=speech+1.5;
    elseif spectral_centroid<spectral_centroid_music_ave
        music=music+1.5;
    end

    
    
    %RESULT: 0 means failed, 0.5 means blurred line, and 1 means success.
    if music<speech
        result_speech(k)=1;
    elseif music>speech
        result_speech(k)=0;
    else
        result_speech(k)=0.5;        
    end
    
end

accuracy_music=(sum(result_music(:) == 1))/num_files
accuracy_speech=(sum(result_speech(:) == 1))/num_files

disp('Algorithm Evaluated');

%}

%% Plots: %%

%Time Domain:

        %Silence:
        figure(1)
        subplot(2,2,1)
        plot(prop_silence_music,'b');
        hold on
        plot(vectorsilencemusic,'--');
        plot(prop_silence_speech,'r');
        plot(vectorsilencespeech,'--');
        title('Proportion of silence given a certain threshold');
        legend('Music','Music average','Speech','Speech average');
        xlabel('Samples given training data');
        ylabel('Proportion');
        hold off
        
        %Variance:
        variance_music_av_plot=zeros(num_files,1);
        variance_music_av_plot(:)=variances_music_av;
        variance_speech_av_plot=zeros(num_files,1);
        variance_speech_av_plot(:)=variances_speech_av;
        
        subplot(2,2,2)
        plot(variances_music,'b');
        hold on
        plot(variance_music_av_plot,'--');
        plot(variances_speech,'r');
        plot(variance_speech_av_plot,'--');
        title('Variances for Music and Speech based on 39 Signals');
        legend('Music','Music average var','Speech','Speech average var');
        xlabel('# Signal');
        ylabel('Variance');
        hold off
        
        %Zero-Crossing:
        zero_crossing_music_av_plot=zeros(num_files,1);
        zero_crossing_music_av_plot(:)=zero_crossing_music_av;
        zero_crossing_speech_av_plot=zeros(num_files,1);
        zero_crossing_speech_av_plot(:)=zero_crossing_speech_av;
        
        subplot(2,2,3)
        plot(zero_crossing_music,'b');
        hold on
        plot(zero_crossing_music_av_plot,'--');
        plot(zero_crossing_speech,'r');
        plot(zero_crossing_speech_av_plot,'--');
        title('Zero-Crossing (ZC) rate for Music and Speech based on 39 Signals');
        legend('Music','Music average ZC','Speech','Speech average ZC');
        xlabel('# Signal');
        ylabel('Zero-Crossing Rate');
        hold off
        
        %Autocorrelation:
        media_music_autocorr_plot=zeros(num_files,1);
        media_music_autocorr_plot(:)=media_music_autocorr;
        media_speech_autocorr_plot=zeros(num_files,1);
        media_speech_autocorr_plot(:)=media_speech_autocorr;
        
        subplot(2,2,4)
        plot(media_music_autocorr_plot,'--');
        hold on
        plot(media_speech_autocorr_plot,'--');
        title('Average Autocorrelation (AC) for Music and Speech based on 39 Signals');
        legend('Music average AC','Speech average AC');
        xlabel('# Signal');
        ylabel('Autocorrelation');
        hold off
        
        
%Frequency Domain:
        
        figure(2)
        %RMS (Music):
        RMS_music_ave_intervals=zeros(num_files,num_intervals);
        
        for i=1:num_intervals
            RMS_music_ave_intervals(:,i)=RMS_music_ave(i);
        end
        
        subplot(2,2,1)
        plot(RMS_music_ave_intervals(:,1),'--');
        hold on
        plot(RMS_music_ave_intervals(:,2),'--');
        plot(RMS_music_ave_intervals(:,3),'--');
        plot(RMS_music_ave_intervals(:,4),'--');
        plot(RMS_music_ave_intervals(:,5),'--');
        plot(RMS_music_ave_intervals(:,6),'--');
        title('Average RMS for Music, in 6 Spectrum Intervals, based on 39 Signals');
        legend('Interval 1','Interval 2','Interval 3','Interval 4','Interval 5','Interval 6');
        xlabel('# Signal');
        ylabel('RMS');
        hold off
       
        %RMS (Speech):
        RMS_speech_ave_intervals=zeros(num_files,num_intervals);
        
        for i=1:num_intervals
            RMS_speech_ave_intervals(:,i)=RMS_speech_ave(i);
        end
        
        subplot(2,2,2)
        plot(RMS_speech_ave_intervals(:,1),'--');
        hold on
        plot(RMS_speech_ave_intervals(:,2),'--');
        plot(RMS_speech_ave_intervals(:,3),'--');
        plot(RMS_speech_ave_intervals(:,4),'--');
        plot(RMS_speech_ave_intervals(:,5),'--');
        plot(RMS_speech_ave_intervals(:,6),'--');
        title('Average RMS for Speech, in 6 Spectrum Intervals, based on 39 Signals');
        legend('Interval 1','Interval 2','Interval 3','Interval 4','Interval 5','Interval 6');
        xlabel('# Signal');
        ylabel('RMS');
        hold off      
        
        
        %Spectral Centroid:
        sc_music__plot=zeros(num_files,1);
        sc_music__plot(:)=spectral_centroid_music_ave;
        sc_speech__plot=zeros(num_files,1);
        sc_speech__plot(:)=spectral_centroid_speech_ave;
        
        subplot(2,2,3)
        plot(spectral_centroid_music,'b');
        hold on
        plot(sc_music__plot,'--');
        plot(spectral_centroid_speech,'r');
        plot(sc_speech__plot,'--');
        title('Spectral Centroid (SC) for Music and Speech based on 39 Signals');
        legend('Music','Music average SC','Speech','Speech average SC','location','southeast');
        xlabel('# Signal');
        ylabel('Spectral Centroid');
        hold off              

        %Success Rate / Accuracy:
        figure(3)
        aux_plot_music=1:length(result_music);
        aux_plot_speech=1:length(result_speech);
        
        accuracy_music_plot=zeros(num_files,1);
        accuracy_speech_plot=zeros(num_files,1);
        
        accuracy_music_plot(:)=accuracy_music;
        accuracy_speech_plot(:)=accuracy_speech;
        
        subplot(2,1,1)
        scatter(aux_plot_music,result_music,'d')
        hold on
        plot(accuracy_music_plot,'--')
        title('Success Rate for 39 Music Signals (1=Success, 0=Failure)');
        legend('Success Rate','Accuracy for Music (%)','location','southeast')
        xlabel('# Signal');
        ylabel('Success Rate');
        hold off
        
        subplot(2,1,2)
        scatter(aux_plot_speech,result_speech,'d')
        hold on
        plot(accuracy_speech_plot,'--')
        title('Success Rate for 39 Speech Signals (1=Success, 0=Failure)');
        legend('Success Rate','Accuracy for Speech (%)','location','southeast')
        xlabel('# Signal');
        ylabel('Success Rate');
        hold off