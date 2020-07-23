# On the usage of the ECMWF API.

After providing all the material related to the ECMWF token, I decided to have a look to the material provided to you.

I wanted to be sure that the material I provided to you was not obsolete.

## Password reset
I went to https://www.ecmwf.int and I tried to login with esaf/07LZTm that were provided along with the token.

I realised that the password was not valid anymore. I asked Pierre Femenias to reset his account.
The new credentials are :

** esaf / SW0e#`R@ **

However, you shouldn't need them at all now because I managed to get the API access key to make the defaults scripts working.


## Query with the API client
Following instruction in https://github.com/ecmwf/ecmwf-api-client, I retrieved an API
key at https://api.ecmwf.int/v1/key/

This gave me the Following json string that you need to write in ~/.ecmwfapirc:
```
{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "4175ae6a1be3339184f1b3c988b3ff4d",
    "email" : "pierre.femenias@esa.int"
}
```


Then running one of the scripts provided by DLR worked straight away:
```
python3 nwp_request.129.setap.example_20190805T120000
2020-07-23 15:34:21 ECMWF API python library 1.5.4
2020-07-23 15:34:21 ECMWF API at https://api.ecmwf.int/v1
2020-07-23 15:34:22 Welcome Pierre Femenias
2020-07-23 15:34:22 In case of problems, please check https://confluence.ecmwf.int/display/WEBAPI/Web+API+FAQ or contact servicedesk@ecmwf.int
....

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
......
mars - INFO   - 20200723.133639 - No errors reported
Process '['nice', 'mars', '/tmp/20200723-1330/af/tmp-_marsFO6LbO.req']' finished
2020-07-23 15:36:49 Request is complete
2020-07-23 15:36:49 Transfering 12.3677 Mbytes into ECMWF_OPER_ML00_06H_129_GP_N640_20190805T120000
2020-07-23 15:36:49 From https://stream.ecmwf.int/data/webmars-private-svc-green-003/data/scratch/20200723-1330/42/_mars-webmars-private-svc-green-003-6fe5cac1a363ec1525f54343b6cc9fd8-kgtL_f.grib
2020-07-23 15:36:58 Transfer rate 1.37155 Mbytes/s
2020-07-23 15:36:58 Done.
```

## Resource to interpret the query 
Overal MARS documentation here: https://confluence.ecmwf.int/display/UDOC/MARS+user+documentation
The meaning of each  keyword can be found here :  https://confluence.ecmwf.int/display/UDOC/Keywords+in+MARS+and+Dissemination+requests
server.execute(
    {
    "class"   : "od",  #routine operations (od)
    "date"    : "2019-08-05",
    "expver"  : "1",  # Operational data
    "levtype" : "ml",  #Common values are: model level (ml), pressure level (pl), surface (sfc), potential vorticity (pv), potential temperature (pt) and depth (dp)."
    "param"   : "129",
    "levelist": "1",  "model pressure level"
    "stream"  : "oper",
    "grid"    : "0.1/0.1",
    "time"    : "12",
    "type"    : "an",
    "area"    : "90/-180/-90/180",
    },
