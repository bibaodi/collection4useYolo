#include <stdint.h>
#include <stdio.h>

char *msg="In computing, the terms big endian and little endian refer to the order in which a value's bytes are stored in memory: with the most significant (big) or least significant (little) part first.";

int main(void)
{
        uint32_t x = 0x12345678;
	const uint32_t littleEndianValue=0x78;
        int i;
	int j=0;

	j = ((char*)&x)[0];
        printf("j=0x%02x ;\n", j);
	if (littleEndianValue==j){
		printf("LittleEndian\n");
	} else {
		printf("BigEndian\n");
	}

        for (i = 0; i < sizeof(x); i++) {
                printf("mem[%d] = 0x%02x\n", i, ((char*)&x)[i]);
	}

	printf("%s\n", msg);
        return 0;
}
