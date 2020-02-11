'''
Data for learning to execute task.  
'''
import numpy as np



class Expression:
    pass

class Number(Expression):
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)

class BinaryExpression(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return "(" + str(self.left) + self.op + str(self.right) + ")"

def randomExpression(prob, rng):
    p = rng.rand()
    if p > prob:
        return Number(rng.randint(0, 10)), 0
    else:
        left, leftdepth = randomExpression(prob / 1.2, rng)
        op = rng.choice(["+", "-", "*"])
        right, rightdepth = randomExpression(prob / 1.2, rng)
        return (BinaryExpression(left, op, right),
                max(leftdepth, rightdepth) + 1)

def randomExpressionFixedLength(length, rng):
    if length == 0:
        # return Number(rng.randint(10, 100)), 0
        return Number(rng.randint(0, 10)), 0
    else:
        leftlen = rng.randint(0, length)
        rightlen = length - leftlen -1
        left, leftdepth = randomExpressionFixedLength(leftlen, rng)
        op = rng.choice(["+", "-", "*", "%"])
        right, rightdepth = randomExpressionFixedLength(rightlen, rng)
        return (BinaryExpression(left, op, right),
                max(leftdepth, rightdepth) + 1)

def generate(count, require_positive, anslen,
             explen=2, mindepth=0, maxdepth=0, rng=None):
    #print('generate!')
    if rng is None:
        rng = np.random.RandomState()
    if explen == 0:
        Expression = randomExpression
        explen = 1
    else:
        Expression = randomExpressionFixedLength
    if maxdepth == 0:
        maxdepth = explen

    exps = []
    i = 0
    while i < count:
        #print('i', i, 'explen', len(expvalue), 'target len', target_anslen)
        exp, depth = Expression(explen, rng=rng)
        exp = exp.__str__()
        try:
            expvalue = str(eval(exp))
        except ZeroDivisionError:
            continue

        is_negative = True if expvalue[0] == '-' else False
        if require_positive and is_negative:
            continue
        
        
        target_anslen = anslen + (1 if is_negative else 0)
        #print('i', i, 'explen', len(expvalue), 'target len', target_anslen)
        if len(expvalue) == target_anslen and \
                                    depth <= maxdepth and depth >= mindepth:
            exps.append([exp, expvalue])
            i += 1

    return exps


class setup_math_expressions(object):
    def __init__(self, n_valid, n_test, expression_length=3,
                 answer_length=1, sequential_answer=False,
                 require_positive=True, rng=None):
        self.n_valid = n_valid
        self.n_test = n_test
        self.expression_length = expression_length
        self.answer_length = answer_length
        self.sequential_answer = sequential_answer
        self.require_positive = require_positive
        self.rng = rng if rng is not None else np.random.RandomState()
        
        #print('state setup')

        # Embedding indices for all possible characters.
        self._vocabulary = ['0','1','2','3','4','5','6','7','8','9',
                            '(',')','+','-','*','%','=','_']
        self._char_dict = dict(zip(self._vocabulary,
                                   range(len(self._vocabulary))))
        
        # Pre-generate validation and test sets. Using same rng.
        self._hash_map = {}
        self._validation_set = self._generate_unique(n_valid)
        self._testing_set = self._generate_unique(n_test)
        
    def _generate_unique(self, num, with_replacement=False):
        #print('gen unique')
        arr_exp, arr_ans = [], []
        accumulated = 0
        while accumulated < num:
            #print('acc', accumulated)
            exp, ans = generate(count=1,
                                 require_positive=self.require_positive,
                                 anslen=self.answer_length,
                                 explen=self.expression_length,
                                 rng=self.rng)[0]
         
            #print('exp', exp)
            #print('ans', ans)

            # Add marker to expression signifying start of answer.
            # Extend expression to have timesteps for each answer step.
            n = self.answer_length+1 if self.sequential_answer else 1
            exp += '='+'_'*n
            
            exp_hash = hash(exp)
            if exp_hash not in self._hash_map:
                if not with_replacement:
                    self._hash_map[exp_hash] = [exp, ans]
                
                # Convert to model-interpretable form.
                exp = [self._char_dict[char] for char in exp]
                if self.sequential_answer:
                    # Symbolic form.
                    if ans[0]!='-':
                        ans = '+{}'.format(ans)
                    ans = [self._char_dict[char] for char in ans]
                arr_exp.append(exp)
                arr_ans.append(ans)
                accumulated += 1
        arr_exp = np.array(arr_exp, dtype=np.int32)
        arr_ans = np.array(arr_ans, dtype=np.int32)
        if not self.sequential_answer:
            arr_ans = arr_ans[:,None]   # expand dim
        return arr_exp, arr_ans
    
    def get_validation(self):
        return self._validation_set
    
    def get_testing(self):
        return self._testing_set
    
    def get_training_batch(self, batch_size):
        x, y = self._generate_unique(batch_size, with_replacement=True)
        batch = {'input': x,
                 'target': y}
        return batch
    
    def get_vocabulary(self):
        return self._vocabulary
 
n_valid = 64
n_test = 11
sequential_answer=True
require_positive = False

#data = setup_math_expressions(n_valid=n_valid,
#                                  n_test=n_test,
#                                  expression_length=expression_length,
#                                  answer_length=answer_length,
#                                  sequential_answer=sequential_answer,
#                                  require_positive=require_positive,
#                                  rng=None)

#data_valid = data.get_validation()
#data_test  = data.get_testing()
#vocabulary = data.get_vocabulary()
#max_len_all = len(data_valid['input'][0])
#max_len_ans = len(data_valid['target'][0])

#print('vocab', vocabulary)

#print('data_valid', data_valid[0].shape, data_valid[1].shape)

#print('min max', data_valid[0][0].min(), data_valid[0][0].max())


def execution_data(expression_length, answer_length):

    data = setup_math_expressions(n_valid=n_valid,
                n_test=n_test,
                expression_length=expression_length,
                answer_length=answer_length,
                sequential_answer=sequential_answer,
                require_positive=require_positive,
                rng=None)


    data_valid = data.get_validation()

    x = data_valid[0]
    y = data_valid[1]

    x_pad = np.zeros_like(y) + 20
    y_pad = np.zeros_like(x) + 20

    x = np.concatenate([x,x_pad],1)
    y = np.concatenate([y_pad,y],1)

    x = np.swapaxes(x, 0,1).astype('int64')
    y = np.swapaxes(y, 0,1).astype('int64')

    return x,y


#for k in range(0,3):
#    x,y = execution_data()

#    print('xs', x.shape)
#    print(x[:,0], y[:,0])

if __name__ == "__main__":

    print('main func')
    
    exp_len = 4
    print('exp len', exp_len)
    x,y = execution_data(exp_len,1)
    print('xy shapes', x.shape, y.shape)

    print(x[:,0], y[:,0])



