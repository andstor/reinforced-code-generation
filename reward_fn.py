from antlr4 import InputStream, CommonTokenStream
from python3parser.python3.Python3Lexer import Python3Lexer
from python3parser.python3.Python3Parser import Python3Parser
from python3parser.python3.Python3ParserVisitor import Python3ParserVisitor
from python3parser.python3.Python3ParserListener import Python3ParserListener
from utils import exponential_decay, normalize

class Listener(Python3ParserListener):
        num_stmt = 0

        def reset(self):
            self.num_stmt = 0

        def enterEveryRule(self, ctx):
            #print(ctx.getText())
            pass
        
        def enterStmt(self, node):
            self.num_stmt += 1
            #print(node.getText())
            


class RewardFn:

    def __init__(self):
        self.factor = 100
        self.avg_tokens = None
        self.avg_errors = None

        input_stream = InputStream("")
        self.lexer = Python3Lexer(input_stream)
        self.lexer.removeErrorListeners() # remove default error listener
        self.token_stream = CommonTokenStream(self.lexer)
        self.parser = Python3Parser(self.token_stream)
        self.parser.removeErrorListeners()
        
        #self.listener = Listener()
        #self.parser.addParseListener(self.listener)

    def __call__(self, prompt, output):

        #self.listener.reset()
        num_tokens = len(self.getSymbolicNames(output))
        norm_tokens = normalize(num_tokens, 0, self.avg_tokens)
        
        try:
            tree = self.parse(prompt + output)
        except Exception as e:
            print(e)
            return 0.0
        num_errors = self.parser.getNumberOfSyntaxErrors()
        norm_errors = normalize(num_errors, 0, self.avg_errors)
        #print("num_errors: ", num_errors)
        #print("norm_errors: ", norm_errors)
        #num_stmt = self.listener.num_stmt
        #lines = len(code.splitlines())
        
        #print("num_tokens: ", num_tokens)
        #print("norm_tokens: ", norm_tokens)

        #num_chars = len(text)
        #norm_chars = normalize(num_chars, 0, self.avg_tokens)
        # clamp to 0.0 - 1.0
        #print("num_chars: ", num_chars)
        #print("clipped_norm_num_chars: ", clipped_norm_num_chars)

        minmax = lambda x, min_value, max_value: max(min(x, max_value), min_value)
        clipped_norm_num_chars = minmax(norm_tokens, 0.00001, 0.99999)

        score = exponential_decay(norm_errors, clipped_norm_num_chars, self.factor)
        #print("score: ", score)
        return minmax(score, 0.0, 1.0)
    
    def calibrate(self, prompts, outputs):
        num_tokens = 0
        num_errors = 0
        #num_stmt = 0
        
        print("calibrating...")
        for prompt, output in zip(prompts, outputs):
            sample = prompt + output
            tree = self.parse(sample)
            num_errors += self.parser.getNumberOfSyntaxErrors()
            #num_stmt += self.listener.num_stmt
            num_tokens += len(self.getSymbolicNames(output)) # Todo replace with self.lexer.reset(); len(self.lexer.getAllTokens());

        self.avg_tokens = max((num_tokens / len(outputs)), 1)
        self.avg_errors = max(num_errors / len(outputs), 1)
        #self.avg_stmt = max(num_stmt / len(outputs), 1)

        print("avg_tokens: ", self.avg_tokens)
        print("avg_errors: ", self.avg_errors)

    def getSymbolicNames(self, text):
        input_stream = InputStream(text)
        self.lexer.inputStream = input_stream
        token_types = [self.lexer.symbolicNames[t.type] for t in self.lexer.getAllTokens()]
        return token_types

    def parse(self, text):
        # TODO: check thread safety
        input_stream = InputStream(text)
        self.lexer.inputStream = input_stream
        self.token_stream.setTokenSource(self.lexer)
        #self.parser.setTokenStream(self.token_stream)
        #self.token_stream.reset()
        self.parser.reset()
        
        
        tree = self.parser.file_input()
        #print(tree.toStringTree(recog=self.parser))


        return tree
    
if __name__ == "__main__":
    reward_fn = RewardFn()
    print(reward_fn.calibrate(["def asd(:\n    a = 1\n"],["def asd(:\n    a = 1\n"]))