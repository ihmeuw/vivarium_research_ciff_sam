@echo off
NET USE h: /DELETE
NET USE i: /DELETE
NET USE j: /DELETE
NET USE k: /DELETE

NET USE h: \\ihme.washington.edu\IHME\HOMES\%username%
NET USE i: \\ihme.washington.edu\IHME\IHME
NET USE j: \\ihme.washington.edu\IHME\snfs
NET USE k: \\ihme.washington.edu\IHME\cc_resources

:: EXIT