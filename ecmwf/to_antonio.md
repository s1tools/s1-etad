# On the usage of the ECMWF API.

After providing all the material related to the ECMWF token, I decided to have a look
such that I have confidence on how obsolete is the material I provided to Your

## Password reset
I went to https://www.ecmwf.int and I tried to login with esaf/07LZTm.

I realised that the password was not valid. I asked Pierre Femenias to reset his account.
The new credentials are :
esaf / SW0e#`R@

However, you shouldn't need them.


## Query with the API client
Following instruction in https://github.com/ecmwf/ecmwf-api-client, I retrieve a
key at https://api.ecmwf.int/v1/key/

This gave me the Following json string that you need to write in ~/.ecmwfapirc:
{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "4175ae6a1be3339184f1b3c988b3ff4d",
    "email" : "pierre.femenias@esa.int"
}


Then running one of the scripts provided by DLR worked straight away:
python3 nwp_request.129.setap.example_20190805T120000
2020-07-23 15:34:21 ECMWF API python library 1.5.4
2020-07-23 15:34:21 ECMWF API at https://api.ecmwf.int/v1
2020-07-23 15:34:22 Welcome Pierre Femenias
2020-07-23 15:34:22 In case of problems, please check https://confluence.ecmwf.int/display/WEBAPI/Web+API+FAQ or contact servicedesk@ecmwf.int
2020-07-23 15:34:22 Request submitted
2020-07-23 15:34:22 Request id: 5f1991dec7ae2b01d80c44b4
2020-07-23 15:34:22 Request is submitted
2020-07-23 15:34:24 Request is active
Calling 'nice mars /tmp/20200723-1330/af/tmp-_marsFO6LbO.req'
mars - WARN -
mars - WARN - From 29 January 2019 10AM (UTC) MARS uses the interpolation
mars - WARN - provided by the MIR library. For more details, see
mars - WARN - https://confluence.ecmwf.int/display/UDOC/MARS+interpolation+with+MIR
mars - WARN -
MIR environment variables:
MIR_CACHE_PATH=/data/ec_coeff
mars - INFO   - 20200723.133423 - Welcome to MARS
mars - INFO   - 20200723.133423 - MARS Client bundle version: 6.28.6.1
mars - INFO   - 20200723.133423 - MARS Client package version: 6.28.6
mars - INFO   - 20200723.133423 - MARS Client build stamp: 20200717102127
mars - INFO   - 20200723.133423 - MIR version: 1.4.7
mars - INFO   - 20200723.133423 - Using ecCodes version 2.18.0
mars - INFO   - 20200723.133423 - Using odb_api version: 0.15.11 (file format version: 0.5)
mars - INFO   - 20200723.133423 - Using FDB5 version: 5.6.1
mars - INFO   - 20200723.133423 - Maximum retrieval size is 50.00 G
retrieve,stream=oper,levelist=1,area=90/-180/-90/180,levtype=ml,param=129,padding=0,grid=0.1/0.1,expver=1,time=12,date=2019-08-05,type=an,class=odmars - INFO   - 20200723.133424 - Automatic split by date is on

mars - INFO   - 20200723.133424 - Processing request 1

RETRIEVE,
    CLASS      = OD,
    TYPE       = AN,
    STREAM     = OPER,
    EXPVER     = 0001,
    REPRES     = SH,
    LEVTYPE    = ML,
    LEVELIST   = 1,
    PARAM      = 129,
    TIME       = 1200,
    STEP       = 00,
    DOMAIN     = G,
    RESOL      = AUTO,
    AREA       = 90/-180/-90/180,
    GRID       = 0.1/0.1,
    PADDING    = 0,
    DATE       = 20190805

mars - INFO   - 20200723.133424 - Web API request id: 5f1991dec7ae2b01d80c44b4
mars - INFO   - 20200723.133424 - Requesting 1 field
mars - INFO   - 20200723.133424 - dhsbase selecting random server fdb-server-prod3:9000
mars - INFO   - 20200723.133424 - Calling mars on 'fdb-server-prod3', local port is 51565
mars - INFO   - 20200723.133424 - Server task is 863 [FDB5 prod]
mars - INFO   - 20200723.133424 - Retrieving from FDB [FDB5 prod]
mars - INFO   - 20200723.133424 - Retrieving 0 field [FDB5 prod]
mars - INFO   - 20200723.133424 - Looking up FDB indexes: 0.003474 second elapsed, 0 second cpu [FDB5 prod]
mars - INFO   - 20200723.133424 - Calling mars on 'fdb-server-bc', local port is 52677
mars - INFO   - 20200723.133424 - Server task is 501 [FDB BC]
mars - INFO   - 20200723.133424 - Retrieving from FDB [FDB BC]
mars - INFO   - 20200723.133424 - Retrieving 0 field [FDB BC]
mars - INFO   - 20200723.133424 - Looking up FDB indexes: 0.008091 second elapsed, 0 second cpu [FDB BC]
13024 FDB; INFO;   DB$_    Fields DataBase 4.10.0

mars - INFO   - 20200723.133424 - Calling mars on 'marsod', local port is 37486
mars - INFO   - 20200723.133622 - Server task is 376 [marsod]
mars - INFO   - 20200723.133622 - Request cost: 1 field, 3.12953 Mbytes online, nodes: mvr08 [marsod]
mars - INFO   - 20200723.133622 - Transfering 3281554 bytes
mars - INFO   - 20200723.133623 - ShToGridded: loading Legendre coefficients '/data/ec_coeff/mir/legendre/4/local-T1279-GaussianN900-OPT4189816c2e.leg'
mars - INFO   - 20200723.133639 - 1 field retrieved from 'marsod'
mars - INFO   - 20200723.133639 - 1 field has been interpolated
mars - INFO   - 20200723.133639 - Request time:  wall: 2 min 15 sec  cpu: 8 sec
mars - INFO   - 20200723.133639 -   Processing in marsod: wall: 1 min 58 sec
mars - INFO   - 20200723.133639 -   Visiting marsod: wall: 2 min 15 sec
mars - INFO   - 20200723.133639 -   Read from network: 3.13 Mbyte(s) in < 1 sec [20.39 Mbyte/sec]
mars - INFO   - 20200723.133639 -   Post-processing: wall: 16 sec cpu: 8 sec
mars - INFO   - 20200723.133639 -   Writing to target file: 12.37 Mbyte(s) in < 1 sec [140.83 Mbyte/sec]
mars - INFO   - 20200723.133639 - Memory used: 6.00 Gbyte(s)
mars - INFO   - 20200723.133639 - No errors reported
Process '['nice', 'mars', '/tmp/20200723-1330/af/tmp-_marsFO6LbO.req']' finished
2020-07-23 15:36:49 Request is complete
2020-07-23 15:36:49 Transfering 12.3677 Mbytes into ECMWF_OPER_ML00_06H_129_GP_N640_20190805T120000
2020-07-23 15:36:49 From https://stream.ecmwf.int/data/webmars-private-svc-green-003/data/scratch/20200723-1330/42/_mars-webmars-private-svc-green-003-6fe5cac1a363ec1525f54343b6cc9fd8-kgtL_f.grib
2020-07-23 15:36:58 Transfer rate 1.37155 Mbytes/s
2020-07-23 15:36:58 Done.

