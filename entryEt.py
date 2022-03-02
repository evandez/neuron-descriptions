import argparse
def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'

if __name__ == '__main__':
    print('Hello!')
    """
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='minimalexample', type=str, help=None)
    parser.add_argument('--lookup_layer', default='blocks.9.mlp.fc1', type=str, help=None)
    parser.add_argument('--lookup_unit', default=10, type=int, help=None)
    # parser.add_argument('--PROJECT_NAME', default='srd_project', type=str, help=None)
    # parser.add_argument('--GARAM_MASALA', default=100, type=int, help=None)
    # parser.add_argument('--JALAPENO', default=100, type=int, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args

    BOOLS = { # see strbool_description
        'mini_mode': 1,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)
    # parser.add_argument('--debug_toggles', default='0000000',type=str,help='Only string of 0 and 1') 
    # TOGGLES = [parse_bool_from_string(x) for x in args['debug_toggles']]   
    
    kwargs = parser.parse_args()
    dargs = vars(kwargs)  # is a dictionary

    for bkey in BOOLS:
        adjusted_bool = parse_bool_from_string(BOOLS[bkey])
        setattr(kwargs, bkey, adjusted_bool)
        dargs[bkey] = adjusted_bool
        # print(bkey, 'true') if dargs[bkey] else print(bkey, 'false')
        # print(bkey, 'true') if getattr(args,bkey) else print(bkey, 'false')

    # See results here
    # print(dargs)
    # TOGGLES = [parse_bool_from_string(x) for x in dargs['debug_toggles']]
    # print(TOGGLES)

    if dargs['mode'] == 'minimalexample':
        from srcEt.tutorials import runminimalexample
        runminimalexample(dargs)