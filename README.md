# Effects of Uncertainty on Visual Pattern Detection Under Fixed and Variable Noise 


## Summary

This study investigates how signal uncertainty affects the detection of simple and complex visual patterns. Participants completed a two-interval forced-choice task, detecting either sinusoidal gratings or band-limited noise textures. Signal uncertainty was manipulated by presenting either a single signal type or one of five variations per trial. Additionally, noise was either fixed across trials or varied.

Contrast thresholds were measured to assess detection performance under these conditions, providing insight into how uncertainty influences the perception of structured and complex patterns.

## Getting Started


Make sure Docker is running and from the woring director, in cmd, run:

```sh
docker compose build --no-cache
```

And after building, run:

```sh
docker compose up -d
```

Navigate to `https//localhost:5000/dashboard`

## Errors

If you get the error:

`Bind for 0.0.0.0:5000 failed: port is already allocated`

If using mac/linux, run: 

```sh
lsof -i :5000
```
You should see something similar to below:

```sh
COMMAND   PID           USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3   <PID_NUMBER>  user   5u  IPv4  12345      0t0  TCP *:5000 (LISTEN)
```

Note the PID number then run 

```sh
kill <PID_NUMBER>
```

If using Windows, run: 

```sh
netstat -aon | findstr :5000
```
You should see something similar to below:

```sh
TCP    0.0.0.0:5000           0.0.0.0:0              LISTENING       <PID_NUMBER>
```

Note the PID number then run 

```sh
taskkill /PID <PID_NUMBER> /F
```

Ensure that Docker is still running then rerun 

```sh
docker compose up -d
```

## Ending the program


To kill the process, in cmd of the working directory, press:

`Ctrl + c` on keyboard

Or if that doesn't work, similar to above, run:

```sh
netstat -ano | findstr :5000
```

```sh
TCP    127.0.0.1:5000    0.0.0.0:0    LISTENING    <PID_NUMBER>
```

```sh
taskkill /PID <PID_NUMBER> /F
```
