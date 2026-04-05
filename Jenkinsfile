pipeline {
    agent any

    environment {
        ACR_NAME       = 'myliverregistry123'
        ACR_URL        = "${ACR_NAME}.azurecr.io"
        IMAGE_NAME     = 'liver-tumor-api'
        WEBAPP_NAME    = 'liver-tumor-api-prod'
        RG_NAME        = 'LiverML-RG'
        AZURE_CRED_ID  = 'azure-sp-credentials'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Lint') {
            steps {
                sh 'pip install flake8 --quiet && flake8 app/ model/ training/ --max-line-length=120'
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    pip install -r requirements.txt --quiet
                    pytest tests/ -v --tb=short
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${ACR_URL}/${IMAGE_NAME}:${env.BUILD_ID}")
                }
            }
        }

        stage('Push to ACR') {
            steps {
                script {
                    docker.withRegistry("https://${ACR_URL}", 'acr-credentials') {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy to Azure App Service') {
            steps {
                withCredentials([azureServicePrincipal(
                    credentialsId: env.AZURE_CRED_ID,
                    subscriptionIdVariable: 'SUBS_ID',
                    clientIdVariable: 'CLIENT_ID',
                    clientSecretVariable: 'CLIENT_SECRET',
                    tenantIdVariable: 'TENANT_ID'
                )]) {
                    sh """
                        az login --service-principal \
                            -u \$CLIENT_ID -p \$CLIENT_SECRET --tenant \$TENANT_ID

                        az webapp config container set \
                            --name ${WEBAPP_NAME} \
                            --resource-group ${RG_NAME} \
                            --docker-custom-image-name ${ACR_URL}/${IMAGE_NAME}:latest \
                            --docker-registry-server-url https://${ACR_URL}

                        az webapp restart \
                            --name ${WEBAPP_NAME} \
                            --resource-group ${RG_NAME}
                    """
                }
            }
        }

    }

    post {
        success {
            echo "Pipeline succeeded. Image deployed: ${ACR_URL}/${IMAGE_NAME}:${env.BUILD_ID}"
        }
        failure {
            echo "Pipeline failed on stage: ${currentBuild.result}"
        }
    }
}
