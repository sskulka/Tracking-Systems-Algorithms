clc;
clear;
clf;
urlwrite("https://cecas.clemson.edu/ahoover/ece854/labs/magnets-data.txt",...
    "measurement_Q1.txt")
%% Populating true values and measurement data
inp_filename = "measurement_Q1.txt"
fid = fopen(inp_filename); % open the file
iCtr = 0;
y_q1=[];
true_x_q1=[];
true_xdot_q1=[];
while ~feof(fid) % loop over the following until the end of the file is reached.
  line = fgets(fid); % read in one line
  data1 = strsplit(line);
  true_x_q1(end+1) = str2double(data1(1));
  true_xdot_q1(end+1) = str2double(data1(2));
  y_q1(end+1) = str2double(data1(3));
end
true_x_q1=(true_x_q1)';
true_xdot_q1 = (true_xdot_q1)';
y_q1=(y_q1)';
y_q1 = y_q1;
fignum = 1;   

time_q1=(1:1:length(y_q1));
time_q1 = time_q1';
%% Particle Filter Logic
%%number of particles for PF
M = 1000;
Est_pos_arr = [];
%assuming a m*4 particle state array with columns corresponding to x_prev,
%xdot_prev, x_curr, xdot_curr
PF_states = zeros(M,4);

%assuming a m*2 particle weight array with columns corresponding to w_prev, w_curr
PF_weights = zeros(M,2);

%Initially assuming the initial x_prev and xdot_prev values for all particles 
PF_states(:,1) = true_x_q1(1);
PF_states(:,2) = true_xdot_q1(1);

%Initially assuming a uniform 1/M weight for all particles
PF_weights(:,1)=1/M;

%loop the measurement array
for j=1:1:length(y_q1)
    %Calling the state transition equation for every particle
    for i=1:1:M
        [x_next, xdot_next] = state_transition(PF_states(i,1), ...
        PF_states(i,2), time_q1(2)-time_q1(1));

        PF_states(i,3) = x_next;
        PF_states(i,4) = xdot_next;
    end

    %computing the ideal measurement for each particle
    %comparing the ideal measurement for every particle with the sensor reading
    %for that instant -> p(yt|x(m)t)
    %weight update for each of the particles
    weight_sum = 0;
    for i =1:1:M
        PF_weights(i,2) = update_weight(PF_states(i,3),y_q1(j),PF_weights(i,1));
        %weight_sum = weight_sum + PF_weights(i,2);
    end
    weight_sum = sum(PF_weights(:,2));
    %normalize the weight
    for i =1:1:M
        PF_weights(i,2) = PF_weights(i,2)/weight_sum;
    end

    %compute the expected value at that stage
    Exp_val = 0;
    Exp_val = PF_weights(:,2).*PF_states(:,3);
    %Exp_val = (Exp_val) / M;
    Exp_val = sum(Exp_val);
    Est_pos_arr(end+1)=Exp_val;

    %calculating effective sample size ESS = M/(1 + CV)
    CV = 0;
    for i =1:1:M
        CV = CV + (M*PF_weights(i,2)-1)^2;
    end
    CV = CV/M;
    ESS = M/(1+CV);
    
    %checking if resampling is necessary if (ESS < 0.5*M)
    if(ESS < 0.5*M)
        scatter(PF_states(:,1),PF_weights(:,1),5);
        xlim([-6 6])
        ylim([0 0.004])
        xlabel("Weight")
        ylabel("Position")
    
        fprintf("Resampling necessary! at iteration %d\n",j);
        [PF_states(:,3),PF_states(:,4),PF_weights(:,2)] = resample(PF_weights(1:end,2),...
        PF_states(1:end,3),PF_states(1:end,4),M);
    end

    scatter(PF_states(:,1),PF_weights(:,1),5);
    xlim([-6 6])
    ylim([0 0.004])
    xlabel("Weight")
    ylabel("Position")

    %substituting the current states and weights into previous
    PF_weights(:,1) = PF_weights(:,2);
    PF_states(:,1) = PF_states(:,3);
    PF_states(:,2) = PF_states(:,4);
end
%%  Plotting the estimated value, measurement and true state over the course of time
    clf;
    figure(fignum)
    plot(time_q1, Est_pos_arr','.r')
    hold on;
    plot(time_q1, y_q1,'.c')
    hold on;
    plot(time_q1, true_x_q1,'.b')
    ylabel('data (units)');
    xlabel('position (m) ');
    hold on;
    title('Position of Particles wrt time');
    legend('estimated output','measurement value','true position value',...
        'location','northeast');

%% Functions
    function [x_next, xdot_next] = state_transition(x_curr, xdot_curr, T)
    
        sig_a = 0.0625;
        x_next = x_curr + xdot_curr*T;
       
       if x_curr < -20
           xdot_next = 2;
       elseif (-20 <= x_curr) && (x_curr < 0)
           xdot_next = xdot_curr + abs(normrnd(0,sig_a)); 
    
       elseif (0 <= x_curr) && (x_curr <= 20)
           xdot_next = xdot_curr - abs(normrnd(0,sig_a));
       else
           xdot_next = -2;
       end
    end
    
    function [w_next] = update_weight(x, y, w_prev)
        sigm = 4.0;
        sign =  0.003906;
        xm1 = -10;
        xm2 = 10;
    
        %%calculating yt^m
        y_m = (1/(sqrt(2*pi)*sigm))*exp(-(x-xm1)^(2)/(2*sigm*sigm)) ...
            + (1/(sqrt(2*pi)*sigm))*exp(-(x-xm2)^(2)/(2*sigm*sigm));
    
        %computing p(yt|x(m)t)
        probab = (1/(sqrt(2*pi)*sign))*exp(-(y_m-y)^(2)/(2*sign*sign));
    
        %computing weight updation
        w_next = w_prev*probab;
    end
    
    function [New_pos,New_vel,NewW] = resample(PF_weights,PF_states_pos,PF_states_vel,M)
            Q = cumsum(PF_weights);
            t = rand(M+1,1);
            T = sort(t);
            T(M+1) = 1.0;
            i=1;
            j=1;
            while (i<=M)
                if T(i) < Q(j)
                    Index(i)=j;
                    i=i+1;
                else
                    j=j+1;
                end
            end
    
            for i=1:1:M
                New_pos(i)=PF_states_pos(Index(i));
                New_vel(i)=PF_states_vel(Index(i));
                NewW(i)=1/M;
            end
    end